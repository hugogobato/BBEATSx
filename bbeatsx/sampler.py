"""The BBEATSx Gibbs engine (plan §1.2) -- the heart of the project.

A single backfitting MCMC updates the trend, seasonality and generic blocks plus
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

from typing import List, Optional

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
                mu_prior_var=config.errors.sv_mu_prior_var)

        # ---- build blocks
        self.trend_block = self._build_trend()
        self.seasonal_block = self._build_forest_block(
            "seasonal", fs.X_se, config.seasonal.tree_prior, fs.names_se)
        self.generic_block = self._build_forest_block(
            "generic", fs.X_ge, config.generic.resolved_tree_prior(), fs.names_ge)
        self.blocks = [b for b in (self.trend_block, self.seasonal_block,
                                   self.generic_block) if b is not None]

        # ---- prepare (residualize) every block
        for b in self.blocks:
            b.prepare(self.residual)

        # ---- inverse-gamma prior on sigma^2 (homoscedastic path)
        self._a_sigma, self._b_sigma = self._calibrate_sigma_prior()

        # ---- storage for retained draws
        self.sigma2_draws_: List[float] = []        # homoscedastic noise per draw
        self.sigma2_t_draws_: List[np.ndarray] = []  # SV in-sample variance paths
        self.h_last_draws_: List[float] = []         # SV terminal log-variance
        self.current_sigma2 = 1.0
        self.current_sigma2_t = np.ones(self.n)
        self._fitted = False

    # --------------------------------------------------------------- builders
    def _build_trend(self):
        tc = self.config.trend
        if tc.mode in ("linear", "spline"):
            return ConjugateTrendBlock(self.fs.X_tr, self.fs.trend_penalty_cols,
                                       tc.coef_scale, tc.smoothing)
        if tc.mode == "tvp":
            # Map the smoothing knob to a random-walk innovation variance.
            rw_var = (tc.coef_scale ** 2) / max(self.n, 1) * tc.smoothing
            return TVPTrendBlock(self.fs.X_tr, rw_var=max(rw_var, 1e-6))
        # tree mode -> forest foil (expected to flatline on extrapolation)
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
                     keep, gfr)
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
        self._fitted = True
        return self

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
        """Return dict of ``(n, S)`` standardized component prediction arrays."""
        out = {}
        out["trend"] = self.trend_block.in_sample_draws()
        out["seasonal"] = (self.seasonal_block.in_sample_draws()
                           if self.seasonal_block is not None
                           else np.zeros((self.n, self.num_draws)))
        out["generic"] = (self.generic_block.in_sample_draws()
                          if self.generic_block is not None
                          else np.zeros((self.n, self.num_draws)))
        return out
