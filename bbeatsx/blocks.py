"""Additive blocks for the BBEATSx backfitting sampler (plan §1.2).

Every block exposes the same small contract so the Gibbs engine can treat them
uniformly while they all read and write **one shared** :class:`Residual`:

- ``prepare(residual)``           -- initialise and residualise the block.
- ``set_obs_variance(var_t)``     -- inform the block of the current per-obs
  error variance (a scalar in homoscedastic mode, a length-``n`` vector under SV).
- ``sample(residual, gconf, rng, keep, gfr)`` -- draw the block conditional on
  the shared residual, adding its old prediction back and subtracting the new one.
- ``predict_new(features)``       -- per-draw predictions at new rows -> ``(n*, S)``.
- ``in_sample_draws()``           -- per-draw in-sample predictions -> ``(n, S)``.

Three concrete blocks:

* :class:`ForestBlock`        -- wraps a ``stochtree`` forest component
  (seasonal, generic, and the ``tree`` trend foil).
* :class:`ConjugateTrendBlock`-- the default extrapolation-safe trend:
  ``F_tr(t) = phi(t) @ beta`` with a Gaussian (optionally P-spline-penalised)
  prior, drawn in closed form.
* :class:`TVPTrendBlock`       -- a time-varying-coefficient linear trend whose
  amplitudes follow a Gaussian random walk, sampled by forward-filter /
  backward-sample (FFBS).  This is the cross-backend realisation of the
  ``tvp`` variant; the BART-coefficient version of plan §0.3 is a planned
  extension requiring leaf-regression support.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import backend as bk
from .config import TreePrior


def _obs_precision(var_t, n: int) -> np.ndarray:
    """Return a length-``n`` precision vector from a scalar/array variance."""
    if np.isscalar(var_t):
        return np.full(n, 1.0 / float(var_t))
    var = np.asarray(var_t, dtype=float).ravel()
    return 1.0 / var


# =========================================================================
class ForestBlock:
    """A BART ensemble restricted to one feature group, on the shared residual."""

    def __init__(self, name: str, X: np.ndarray, tree_prior: TreePrior,
                 global_config, use_weights: bool, feature_names=None) -> None:
        self.name = name
        self.X = np.ascontiguousarray(np.asarray(X, dtype=float))
        if self.X.ndim == 1:
            self.X = self.X[:, None]
        self.n, self.p = self.X.shape
        self.feature_names = feature_names or [f"{name}_{j}" for j in range(self.p)]
        self.tree_prior = tree_prior
        self.use_weights = use_weights
        self._global_config = global_config

        leaf_scale = tree_prior.resolved_leaf_scale()
        self.dataset = bk.Dataset()
        self.dataset.add_covariates(self.X)
        if use_weights:
            self.dataset.add_variance_weights(np.ones(self.n))
        self.forest_config = bk.ForestModelConfig(
            num_trees=tree_prior.num_trees,
            num_features=self.p,
            num_observations=self.n,
            feature_types=np.zeros(self.p, dtype=int),
            variable_weights=np.full(self.p, 1.0 / self.p),
            alpha=tree_prior.alpha,
            beta=tree_prior.beta,
            min_samples_leaf=tree_prior.min_samples_leaf,
            max_depth=tree_prior.max_depth,
            leaf_model_type=0,
            leaf_model_scale=leaf_scale ** 2,
        )
        self.forest = bk.Forest(tree_prior.num_trees, 1, True)
        self.container = bk.ForestContainer(tree_prior.num_trees, 1, True)
        self.sampler = bk.ForestSampler(self.dataset, global_config, self.forest_config)
        self._leaf_var_model = bk.LeafVarianceModel() if tree_prior.sample_leaf_scale \
            else None

    def prepare(self, residual) -> None:
        self.sampler.prepare_for_sampler(self.dataset, residual, self.forest, 0,
                                         np.zeros(1))

    def set_obs_variance(self, var_t) -> None:
        if self.use_weights:
            # stochtree weight convention: Var = sigma^2 / w with sigma^2 pinned
            # to 1 under SV, so w_t = 1 / var_t.
            var = np.full(self.n, float(var_t)) if np.isscalar(var_t) \
                else np.asarray(var_t, dtype=float).ravel()
            self.dataset.update_variance_weights(1.0 / var)

    def sample(self, residual, global_config, rng, np_rng, keep: bool, gfr: bool,
               num_threads: int = 1) -> None:
        # The forest sampler uses the backend RNG (C++ on stochtree); np_rng is
        # the numpy generator for Python-side blocks and is unused here.
        self.sampler.sample_one_iteration(
            self.container, self.forest, self.dataset, residual, rng,
            global_config, self.forest_config, keep, gfr, num_threads)
        if keep and self._leaf_var_model is not None:
            tau = self._leaf_var_model.sample_one_iteration(self.forest, rng, 1.0, 1.0)
            self.forest_config.update_leaf_model_scale(tau)

    def predict_new(self, X_new: np.ndarray) -> np.ndarray:
        ds = bk.Dataset()
        Xn = np.asarray(X_new, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn[:, None]
        ds.add_covariates(np.ascontiguousarray(Xn))
        return np.atleast_2d(self.container.predict(ds))

    def in_sample_draws(self) -> np.ndarray:
        return np.atleast_2d(self.container.predict(self.dataset))

    def predict_single(self, X_new: np.ndarray, draw: int) -> np.ndarray:
        """Prediction of a single retained draw ``draw`` at new rows ``(n*,)``."""
        ds = bk.Dataset()
        Xn = np.asarray(X_new, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn[:, None]
        ds.add_covariates(np.ascontiguousarray(Xn))
        out = self.container.predict_raw_single_forest(ds, draw)
        return np.asarray(out).ravel()

    def current_prediction(self) -> np.ndarray:
        return self.forest.predict(self.dataset)

    def split_counts(self) -> np.ndarray:
        return self.container.get_overall_split_counts(self.p)

    def split_counts_per_draw(self) -> np.ndarray:
        """Posterior of per-feature split counts -> ``(S, p)`` (plan §0.4)."""
        S = self.container.num_samples()
        out = np.zeros((S, self.p))
        for si in range(S):
            out[si] = self.container.get_forest_split_counts(si, self.p)
        return out


# =========================================================================
class ConjugateTrendBlock:
    """Extrapolation-safe parametric trend ``F_tr(t) = phi(t) @ beta`` (§3.6).

    The coefficient prior is Gaussian, ``beta ~ N(0, S0)``, with
    ``S0^{-1} = coef_scale^{-2} I + smoothing * D2' D2`` where ``D2`` is the
    2nd-difference operator restricted to spline columns (a P-spline / random-walk
    smoothing prior).  The full conditional ``beta | r, sigma_t^2`` is Gaussian and
    drawn in closed form.
    """

    def __init__(self, Phi: np.ndarray, penalty_mask: np.ndarray,
                 coef_scale: float, smoothing: float) -> None:
        self.Phi = np.ascontiguousarray(np.asarray(Phi, dtype=float))
        if self.Phi.ndim == 1:
            self.Phi = self.Phi[:, None]
        self.n, self.p = self.Phi.shape
        self.beta = np.zeros(self.p)
        self.draws: list = []

        prior_prec = np.eye(self.p) / (coef_scale ** 2)
        pen_idx = np.where(np.asarray(penalty_mask, dtype=bool))[0]
        if pen_idx.size >= 3 and smoothing > 0:
            k = pen_idx.size
            D = np.zeros((k - 2, k))
            for i in range(k - 2):
                D[i, i] = 1.0
                D[i, i + 1] = -2.0
                D[i, i + 2] = 1.0
            DtD = D.T @ D
            full = np.zeros((self.p, self.p))
            full[np.ix_(pen_idx, pen_idx)] = DtD
            prior_prec = prior_prec + smoothing * full
        self.prior_prec = prior_prec

    def prepare(self, residual) -> None:
        # beta starts at zero -> zero prediction -> residual unchanged.
        pass

    def set_obs_variance(self, var_t) -> None:
        self._prec = _obs_precision(var_t, self.n)

    def sample(self, residual, global_config, rng, np_rng, keep: bool, gfr: bool,
               num_threads: int = 1) -> None:
        residual.add_vector(self.Phi @ self.beta)        # add back old trend
        # stochtree returns the residual as an (n, 1) column; the numpy backend
        # as (n,). Ravel so downstream linear algebra is backend-independent.
        R = np.asarray(residual.get_residual()).ravel()
        prec = self._prec
        PtW = self.Phi.T * prec                          # (p, n)
        A = PtW @ self.Phi + self.prior_prec
        b = PtW @ R
        L = np.linalg.cholesky(A)
        mean = np.linalg.solve(L.T, np.linalg.solve(L, b))
        z = np_rng.standard_normal(self.p)
        self.beta = mean + np.linalg.solve(L.T, z)       # N(mean, A^{-1})
        residual.subtract_vector(self.Phi @ self.beta)   # subtract new trend
        if keep:
            self.draws.append(self.beta.copy())

    def predict_new(self, Phi_new: np.ndarray) -> np.ndarray:
        Phi_new = np.asarray(Phi_new, dtype=float)
        if Phi_new.ndim == 1:
            Phi_new = Phi_new[:, None]
        B = np.column_stack(self.draws) if self.draws else np.zeros((self.p, 0))
        return Phi_new @ B

    def in_sample_draws(self) -> np.ndarray:
        B = np.column_stack(self.draws) if self.draws else np.zeros((self.p, 0))
        return self.Phi @ B

    def current_prediction(self) -> np.ndarray:
        return self.Phi @ self.beta


# =========================================================================
class TVPTrendBlock:
    """Time-varying-coefficient linear trend with Gaussian random-walk amplitudes.

    State space (per time ``t``):

        beta_t = beta_{t-1} + eta_t,   eta_t ~ N(0, W),  W = rw_var * I
        r_t    = phi(t)' beta_t + eps_t, eps_t ~ N(0, var_t)

    Sampled by FFBS.  Forecasting rolls ``beta_T`` forward under the random walk,
    so the extrapolation is linear-in-``phi(t)`` (extrapolation-safe) while the
    amplitudes -- and their uncertainty -- evolve.
    """

    def __init__(self, Phi: np.ndarray, rw_var: float, init_var: float = 100.0) -> None:
        self.Phi = np.ascontiguousarray(np.asarray(Phi, dtype=float))
        if self.Phi.ndim == 1:
            self.Phi = self.Phi[:, None]
        self.n, self.d = self.Phi.shape
        self.rw_var = float(rw_var)
        self.W = np.eye(self.d) * self.rw_var
        self.m0 = np.zeros(self.d)
        self.C0 = np.eye(self.d) * init_var
        self.beta_path = np.zeros((self.n, self.d))
        self.beta_T_draws: list = []     # terminal state per kept draw (for forecast)
        self.path_draws: list = []       # full in-sample path per kept draw

    def _current_pred(self) -> np.ndarray:
        return np.einsum("td,td->t", self.Phi, self.beta_path)

    def prepare(self, residual) -> None:
        pass

    def set_obs_variance(self, var_t) -> None:
        self._var = (np.full(self.n, float(var_t)) if np.isscalar(var_t)
                     else np.asarray(var_t, dtype=float).ravel())

    def sample(self, residual, global_config, rng, np_rng, keep: bool, gfr: bool,
               num_threads: int = 1) -> None:
        residual.add_vector(self._current_pred())
        R = np.asarray(residual.get_residual()).ravel()
        gen = np_rng

        # ---- forward filter
        ms = np.zeros((self.n, self.d))
        Cs = np.zeros((self.n, self.d, self.d))
        m, C = self.m0, self.C0
        for t in range(self.n):
            a = m
            Rm = C + self.W
            phi = self.Phi[t]
            f = phi @ a
            Qv = phi @ Rm @ phi + self._var[t]
            gain = Rm @ phi / Qv
            m = a + gain * (R[t] - f)
            C = Rm - np.outer(gain, gain) * Qv
            ms[t] = m
            Cs[t] = C

        # ---- backward sample
        beta = np.zeros((self.n, self.d))
        beta[-1] = gen.multivariate_normal(ms[-1], _sym(Cs[-1]))
        for t in range(self.n - 2, -1, -1):
            Rm = Cs[t] + self.W
            J = Cs[t] @ np.linalg.inv(Rm)
            mean = ms[t] + J @ (beta[t + 1] - ms[t])
            cov = Cs[t] - J @ Rm @ J.T
            beta[t] = gen.multivariate_normal(mean, _sym(cov))
        self.beta_path = beta
        residual.subtract_vector(self._current_pred())
        if keep:
            self.beta_T_draws.append(beta[-1].copy())
            self.path_draws.append(self._current_pred().copy())

    def predict_new(self, Phi_new: np.ndarray, np_rng=None) -> np.ndarray:
        """Roll ``beta_T`` forward under the random walk for each kept draw."""
        Phi_new = np.asarray(Phi_new, dtype=float)
        if Phi_new.ndim == 1:
            Phi_new = Phi_new[:, None]
        h = Phi_new.shape[0]
        S = len(self.beta_T_draws)
        out = np.zeros((h, S))
        gen = np_rng if np_rng is not None else np.random.default_rng(0)
        for s in range(S):
            b = self.beta_T_draws[s].copy()
            for step in range(h):
                b = b + gen.normal(0.0, np.sqrt(self.rw_var), size=self.d)
                out[step, s] = Phi_new[step] @ b
        return out

    def in_sample_draws(self) -> np.ndarray:
        return (np.column_stack(self.path_draws) if self.path_draws
                else np.zeros((self.n, 0)))

    def current_prediction(self) -> np.ndarray:
        return self._current_pred()


def _sym(M: np.ndarray) -> np.ndarray:
    """Symmetrise and nudge a covariance to be positive-definite."""
    M = 0.5 * (M + M.T)
    return M + np.eye(M.shape[0]) * 1e-10
