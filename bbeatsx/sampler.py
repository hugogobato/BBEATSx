"""The BBEATSx Gibbs engine (plan §1.2) -- the heart of the project.

A single backfitting MCMC updates the configured additive mean blocks plus
the error-variance model, all sharing **one** :class:`Residual` and one global
variance state.  Each block, in turn, adds its current prediction back into the
shared residual, draws itself conditional on that partial residual, and subtracts
its new prediction -- exactly the cross-group backfitting of concept §3.3.  After
the mean blocks, the error model draws either a homoscedastic ``sigma^2`` (inverse
gamma) or the SV log-variance path ``h_{1:T}`` (:mod:`bbeatsx.sv`).

The sampler standardises ``y`` internally (BART priors and the leaf scales are
calibrated on the unit-variance scale); :attr:`y_mean_`/:attr:`y_std_` let callers
map draws back to the original scale.

Per retained draw it stores each component's in-sample prediction and the error
state, which the forecaster (:mod:`bbeatsx.forecast`) and interpretability tools
(:mod:`bbeatsx.interpret`) consume.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import chi2

from . import backend as bk
from .blocks import ConjugateTrendBlock, ForestBlock, TVPTrendBlock
from .config import BBEATSxConfig
from .features import FeatureSet
from .sv import SVSampler


class BBEATSxSampler:
    """Run the BBEATSx backfitting MCMC given prebuilt feature matrices."""

    def __init__(self, fs: FeatureSet, config: BBEATSxConfig) -> None:
        self.fs = fs
        self.config = config
        self.backend = bk.BACKEND
        self.backend_version = bk.BACKEND_VERSION

        # ---- standardize the target (priors live on the unit-variance scale)
        y = np.asarray(fs.y, dtype=float).ravel()
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y)) or 1.0
        self.z_ = (y - self.y_mean_) / self.y_std_
        self.n = self.z_.shape[0]

        self.sv_mode = config.errors.mode == "sv"
        # Two RNGs: the backend RNG drives the forest sampler (a C++ generator on
        # the stochtree backend, unusable for numpy draws); np_rng drives the
        # Python-side blocks (conjugate/TVP trend, SV, forecast noise).
        self.rng = bk.RNG(config.mcmc.seed)
        self.np_rng = np.random.default_rng(config.mcmc.seed)

        # ---- shared state
        self.global_config = bk.GlobalModelConfig(1.0)
        self.residual = bk.Residual(self.z_.copy())
        self.global_var_model = bk.GlobalVarianceModel()
        self.sv: Optional[SVSampler] = None
        if self.sv_mode:
            self.sv = SVSampler(
                self.n, phi=config.errors.sv_phi, sigma_h=config.errors.sv_sigma_h,
                mu_prior_var=config.errors.sv_mu_prior_var,
                exact=getattr(config.errors, "sv_exact", True))

        # ---- build blocks
        self.trend_block = self._build_trend()
        self.seasonal_block = self._build_forest_block(
            "seasonal", fs.X_se, config.seasonal.tree_prior, fs.names_se)
        self.generic_block = None
        self.exogenous_block = None
        self.autoregressive_block = None
        if config.generic.component_layout == "combined":
            self.generic_block = self._build_forest_block(
                "generic", fs.X_ge, config.generic.resolved_tree_prior(), fs.names_ge)
            self.atomic_component_names = ("trend", "seasonal", "generic")
        elif config.generic.component_layout == "split":
            priors = config.generic.resolved_component_priors(
                has_exogenous=fs.X_ex.shape[1] > 0,
                has_autoregressive=fs.X_ar.shape[1] > 0,
            )
            if "exogenous" in priors:
                self.exogenous_block = self._build_forest_block(
                    "exogenous", fs.X_ex, priors["exogenous"], fs.names_ex)
            if "autoregressive" in priors:
                self.autoregressive_block = self._build_forest_block(
                    "autoregressive", fs.X_ar, priors["autoregressive"], fs.names_ar)
            self.atomic_component_names = (
                "trend", "seasonal", "exogenous", "autoregressive")
        else:
            raise ValueError(
                "generic.component_layout must be 'combined' or 'split', got "
                f"{config.generic.component_layout!r}"
            )

        self.component_blocks = {
            "trend": self.trend_block,
            "seasonal": self.seasonal_block,
            "generic": self.generic_block,
            "exogenous": self.exogenous_block,
            "autoregressive": self.autoregressive_block,
        }
        # Systematic scan: structural blocks first, history/state correction last.
        self.blocks = [self.component_blocks[name]
                       for name in self.atomic_component_names
                       if self.component_blocks[name] is not None]

        # ---- prepare (residualize) every block
        for b in self.blocks:
            b.prepare(self.residual)

        # ---- inverse-gamma prior on sigma^2 (homoscedastic path)
        self._a_sigma, self._b_sigma = self._calibrate_sigma_prior()

        # ---- storage for retained draws
        self.sigma2_draws_: List[float] = []        # homoscedastic noise per draw
        self.sigma2_t_draws_: List[np.ndarray] = []  # SV in-sample variance paths
        self.h_last_draws_: List[float] = []         # SV terminal log-variance
        self.mu_draws_: List[float] = []             # SV level per draw (F-2.4-1)
        self.current_sigma2 = 1.0
        self.current_sigma2_t = np.ones(self.n)
        self._fitted = False

    # --------------------------------------------------------------- builders
    def _build_trend(self):
        tc = self.config.trend
        if tc.mode in ("linear", "spline"):
            if tc.mode == "linear" and tc.degree >= 2:
                warnings.warn(
                    f"trend mode='linear' with degree={tc.degree} >= 2 amplifies "
                    "extrapolation variance as (1+delta)^(2*degree) (WP-2.3 "
                    "Thm 2.3.7 remark ii); degree=1 is recommended unless the "
                    "truth's tail is known polynomial.", stacklevel=2)
            # De-sloping (fix F2) only bites on the rank-deficient spline design.
            deslope = tc.deslope and tc.mode == "spline"
            return ConjugateTrendBlock(self.fs.X_tr, self.fs.trend_penalty_cols,
                                       tc.coef_scale, tc.smoothing, deslope=deslope)
        if tc.mode == "tvp":
            # Map the smoothing knob to a random-walk innovation variance (the
            # prior mean when the rw_var hyperprior is learned).
            rw_var = (tc.coef_scale ** 2) / max(self.n, 1) * tc.smoothing
            return TVPTrendBlock(self.fs.X_tr, rw_var=max(rw_var, 1e-6),
                                 learn_rw_var=tc.learn_rw_var,
                                 prior_a=tc.rw_var_prior_a)
        # tree mode -> forest foil; its forecast trend is constant in the horizon
        # by construction (WP-2.3 Lemma 2.3.2): an ablation foil, not a forecaster.
        warnings.warn(
            "trend mode='tree' is the extrapolation-failure foil: its forecast "
            "trend is exactly constant in the horizon (WP-2.3 Lemma 2.3.2 / "
            "Thm 2.3.3) and bands do not widen. Use 'spline'/'linear'/'tvp' for "
            "real forecasting.", stacklevel=2)
        return ForestBlock("trend", self.fs.X_tr, tc.tree_prior,
                           self.global_config, self.sv_mode, self.fs.names_tr)

    def _build_forest_block(self, name, X, tree_prior, names):
        if X is None or X.shape[1] == 0:
            return None
        return ForestBlock(name, X, tree_prior, self.global_config,
                           self.sv_mode, names)

    def _calibrate_sigma_prior(self):
        """Standard BART inverse-gamma calibration on the standardized scale."""
        nu = self.config.errors.nu
        q = self.config.errors.q
        # Overdispersed variance estimate from a quick least-squares fit.
        cols = [m for m in (self.fs.X_tr, self.fs.X_se, self.fs.X_ge)
                if m is not None and m.shape[1] > 0]
        X = np.column_stack(cols) if cols else np.ones((self.n, 1))
        try:
            beta, *_ = np.linalg.lstsq(X, self.z_, rcond=None)
            resid = self.z_ - X @ beta
            sigma2_hat = max(float(np.var(resid)), 1e-3)
        except Exception:
            sigma2_hat = 1.0
        lam = sigma2_hat * chi2.ppf(1.0 - q, nu) / nu
        return nu / 2.0, nu * lam / 2.0

    # ------------------------------------------------------------------ engine
    def _set_obs_variance(self) -> None:
        var = self.current_sigma2_t if self.sv_mode else self.current_sigma2
        for b in self.blocks:
            b.set_obs_variance(var)

    def _sweep(self, gfr: bool, keep: bool) -> None:
        self._set_obs_variance()
        for b in self.blocks:
            b.sample(self.residual, self.global_config, self.rng, self.np_rng,
                     keep, gfr, self.config.mcmc.num_threads)
        # error model on the full residual r = z - sum(F_c)
        if self.sv_mode:
            eps = np.asarray(self.residual.get_residual()).ravel()
            self.current_sigma2_t = self.sv.step(eps, self.np_rng)
            self.global_config.update_global_error_variance(1.0)
        else:
            s2 = self.global_var_model.sample_one_iteration(
                self.residual, self.rng, self._a_sigma, self._b_sigma)
            self.current_sigma2 = float(s2)
            self.global_config.update_global_error_variance(self.current_sigma2)
        if keep:
            if self.sv_mode:
                self.sigma2_t_draws_.append(self.current_sigma2_t.copy())
                self.h_last_draws_.append(float(self.sv.h[-1]))
                self.mu_draws_.append(float(self.sv.mu))
            else:
                self.sigma2_draws_.append(self.current_sigma2)

    def run(self) -> "BBEATSxSampler":
        mc = self.config.mcmc
        for _ in range(mc.num_gfr):
            self._sweep(gfr=True, keep=False)
        for _ in range(mc.num_burnin):
            self._sweep(gfr=False, keep=False)
        for i in range(mc.num_mcmc):
            self._sweep(gfr=False, keep=(i % mc.thin == 0))
        # De-sloping map varsigma on the conjugate spline trend (WP-2.3 fix F2):
        # in-sample-invariant, restores extrapolation slope-consistency.
        if isinstance(self.trend_block, ConjugateTrendBlock):
            self.trend_block.apply_deslope()
        self._fix_level_gauge()
        self._fitted = True
        return self

    # ------------------------------------------------------------ level gauge
    def _fix_level_gauge(self) -> None:
        """Fix the additive-level gauge (config ``level_gauge``).

        The additive model identifies the component sum but not how a constant
        is split among its blocks.  For every retained draw, subtract each
        non-trend block's in-sample mean and add all of those offsets to the
        trend.  This handles either the legacy three-block layout or the split
        trend/seasonal/exogenous/autoregressive layout.  The sum is unchanged
        draw-by-draw, so the posterior predictive is untouched; the same
        in-sample offsets are reused out of sample (:mod:`bbeatsx.forecast`).

        Note this constrains the forest *output*, which is what identifiability
        requires; centering the seasonal *design matrix*
        (:attr:`SeasonalConfig.sum_to_zero`) does not, because a sum-of-trees on
        centered features still has an arbitrary output level from its leaves.
        """
        S = self.num_draws
        zero = np.zeros(S)
        deviation_names = self.atomic_component_names[1:]
        if self.config.level_gauge != "trend" or S == 0:
            self.level_offsets_ = {name: zero.copy() for name in deviation_names}
            return

        def _block_mean(block) -> np.ndarray:
            if block is None:
                return np.zeros(S)
            draws = np.asarray(block.in_sample_draws(), dtype=float)
            if draws.size == 0 or draws.shape[1] != S:
                return np.zeros(S)
            return draws.mean(axis=0)

        self.level_offsets_ = {
            name: _block_mean(self.component_blocks[name])
            for name in deviation_names
        }

    def level_offsets(self) -> Dict[str, np.ndarray]:
        """Per-draw ``(S,)`` level offsets moved into the trend by the gauge."""
        off = getattr(self, "level_offsets_", None)
        if off is None:  # sampler fitted before the gauge existed
            z = np.zeros(self.num_draws)
            return {name: z.copy() for name in self.atomic_component_names[1:]}
        return off

    # ----------------------------------------------------------- diagnostics
    def backfitting_residual_error(self) -> float:
        """Max abs of ``z - sum_c F_c(current) - r`` (should be ~0 each sweep)."""
        total = np.zeros(self.n)
        for b in self.blocks:
            total += np.asarray(b.current_prediction()).ravel()
        r = np.asarray(self.residual.get_residual()).ravel()
        return float(np.max(np.abs(self.z_ - total - r)))

    # --------------------------------------------------- retained draw access
    @property
    def num_draws(self) -> int:
        return (len(self.sigma2_t_draws_) if self.sv_mode
                else len(self.sigma2_draws_))

    def in_sample_components(self):
        """Return dict of ``(n, S)`` standardized component prediction arrays.

        The arrays are reported in the gauge selected by ``config.level_gauge``
        (see :meth:`_fix_level_gauge`); under the default ``"trend"`` gauge the
        seasonal and generic arrays are exactly mean-zero over the training rows
        and the trend carries the level.  The row sum is gauge-invariant.
        """
        out = {
            "trend": np.asarray(self.trend_block.in_sample_draws(), dtype=float)
        }
        for name in self.atomic_component_names[1:]:
            block = self.component_blocks[name]
            out[name] = (np.asarray(block.in_sample_draws(), dtype=float)
                         if block is not None
                         else np.zeros((self.n, self.num_draws)))
        off = self.level_offsets()
        out["trend"] = out["trend"] + sum(
            (off[name] for name in self.atomic_component_names[1:]),
            np.zeros(self.num_draws),
        )
        for name in self.atomic_component_names[1:]:
            out[name] = out[name] - off[name]
        return out
