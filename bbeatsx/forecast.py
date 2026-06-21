"""Multi-step forecasting for BBEATSx (plan §1.3, concept §3.4).

Forecasting is **recursive predictive simulation** (BAVART-validated, plan §0.5):
for every retained posterior draw the trend and seasonal contributions -- which
depend only on known-into-the-future features -- are evaluated for the whole
horizon up front, while the generic block is rolled forward one step at a time so
that future AR lags are themselves sampled quantities.  The observation noise is
drawn from the per-draw error state (a homoscedastic ``sigma^2`` or an SV path
propagated by its AR(1) law), so predictive uncertainty propagates honestly with
the horizon.

Every draw of every component is retained, so the result carries **per-component
posterior bands** (trend / seasonal / generic) -- the property no post-hoc method
delivers (concept §2.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .blocks import ConjugateTrendBlock, ForestBlock, TVPTrendBlock
from .features import FeatureBuilder
from .sampler import BBEATSxSampler


@dataclass
class ForecastResult:
    """Posterior-predictive forecast with per-component decomposition.

    All arrays are on the **original** data scale.  ``samples`` and the component
    arrays are ``(H, S)`` (horizon x retained draws).
    """

    t_index: np.ndarray            # absolute integer time index of each step
    samples: np.ndarray            # (H, S) full predictive paths
    components: Dict[str, np.ndarray]  # 'trend'/'seasonal'/'generic' -> (H, S)
    backend: str

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=1)

    def median(self) -> np.ndarray:
        return np.median(self.samples, axis=1)

    def quantile(self, q) -> np.ndarray:
        return np.quantile(self.samples, q, axis=1)

    def interval(self, level: float = 0.9):
        """Return ``(lower, upper)`` central predictive interval at ``level``."""
        a = (1.0 - level) / 2.0
        return self.quantile(a), self.quantile(1.0 - a)

    def component_mean(self, name: str) -> np.ndarray:
        return self.components[name].mean(axis=1)

    def component_interval(self, name: str, level: float = 0.9):
        a = (1.0 - level) / 2.0
        comp = self.components[name]
        return np.quantile(comp, a, axis=1), np.quantile(comp, 1.0 - a, axis=1)


class Forecaster:
    """Recursive posterior-predictive forecaster over a fitted sampler."""

    def __init__(self, sampler: BBEATSxSampler, builder: FeatureBuilder,
                 y_full: np.ndarray) -> None:
        self.s = sampler
        self.b = builder
        self.y_full = np.asarray(y_full, dtype=float).ravel()
        self.n_full = self.y_full.shape[0]

    def forecast(
        self,
        horizon: int,
        exog_future: Optional[Dict[str, np.ndarray]] = None,
        time_future=None,
    ) -> ForecastResult:
        s = self.s
        S = s.num_draws
        if S == 0:
            raise RuntimeError("sampler has no retained draws; call run() first")
        H = int(horizon)
        mean, std = s.y_mean_, s.y_std_
        gen = s.np_rng                      # numpy generator (backend-independent)

        t_future = np.arange(self.n_full, self.n_full + H)

        # ---- trend + seasonal contributions for the whole horizon (no recursion)
        trend_std = self._trend_future(t_future, time_future)        # (H, S)
        seasonal_std = self._seasonal_future(t_future, time_future)  # (H, S)

        # ---- observation-noise variance per (step, draw)
        noise_var = self._noise_variance(H, S, gen)                  # (H, S)

        # ---- recursive roll-forward of the generic block
        generic_std = np.zeros((H, S))
        paths = np.zeros((H, S))
        has_generic = s.generic_block is not None

        for si in range(S):
            hist = np.empty(self.n_full + H)
            hist[: self.n_full] = self.y_full
            for h in range(H):
                t_raw = self.n_full + h
                g = 0.0
                if has_generic:
                    exog_row = self._exog_row(exog_future, h)
                    x_ge = self.b.future_generic_row(t_raw, hist, exog_row)
                    g = float(s.generic_block.predict_single(x_ge, si)[0])
                generic_std[h, si] = g
                eps = gen.normal(0.0, np.sqrt(noise_var[h, si]))
                z_tilde = trend_std[h, si] + seasonal_std[h, si] + g + eps
                y_tilde = mean + std * z_tilde
                hist[t_raw] = y_tilde
                paths[h, si] = y_tilde

        components = {
            # trend carries the overall level (the standardization mean).
            "trend": mean + std * trend_std,
            "seasonal": std * seasonal_std,
            "generic": std * generic_std,
        }
        return ForecastResult(t_index=t_future, samples=paths,
                              components=components, backend=s.backend)

    # ------------------------------------------------------------- internals
    def _trend_future(self, t_future, time_future) -> np.ndarray:
        block = self.s.trend_block
        if isinstance(block, ConjugateTrendBlock):
            Phi = self.b.future_trend_design(t_future)
            return block.predict_new(Phi)
        if isinstance(block, TVPTrendBlock):
            Phi = self.b.future_trend_design(t_future)
            return block.predict_new(Phi, self.s.np_rng)
        # tree-trend foil: forest predicts on engineered t-features (flatlines).
        X_tr = self.b.future_trend_design(t_future)
        return block.predict_new(X_tr)

    def _seasonal_future(self, t_future, time_future) -> np.ndarray:
        if self.s.seasonal_block is None:
            return np.zeros((len(t_future), self.s.num_draws))
        X_se = self.b.future_seasonal_design(t_future, time_future)
        if X_se.shape[1] == 0:
            return np.zeros((len(t_future), self.s.num_draws))
        return self.s.seasonal_block.predict_new(X_se)

    def _noise_variance(self, H, S, gen) -> np.ndarray:
        if self.s.sv_mode:
            out = np.zeros((H, S))
            for si in range(S):
                out[:, si] = self.s.sv.forecast_path(
                    self.s.h_last_draws_[si], H, gen)
            return out
        sig2 = np.asarray(self.s.sigma2_draws_, dtype=float)  # (S,)
        return np.tile(sig2[None, :], (H, 1))

    @staticmethod
    def _exog_row(exog_future, h) -> Optional[Dict[str, float]]:
        if not exog_future:
            return None
        return {k: float(np.asarray(v)[h]) for k, v in exog_future.items()}

    # ===================================================================== #
    # Re-anchored forecasting with FROZEN posterior draws (no refit).        #
    # Used by the periodic-recalibration protocol (real_life_application_v2):#
    # fit once per block, then roll the forecast origin across the block by  #
    # conditioning on the realised series up to each origin.                 #
    # ===================================================================== #
    def forecast_from_origin(
        self,
        y_history: np.ndarray,
        horizon: int,
        exog_future: Optional[Dict[str, np.ndarray]] = None,
        exog_history: Optional[Dict[str, np.ndarray]] = None,
        time_future=None,
        time_history=None,
        sv_filter_window: int = 256,
    ) -> ForecastResult:
        """Forecast ``horizon`` steps from the end of ``y_history`` with the
        parameters sampled at fit time (no resampling).

        ``y_history`` is the **raw** series from the training start up to the last
        observation before the forecast origin (length defines the anchor; it must
        be at least as long as the fitted series).  Trend/seasonal/generic blocks
        are evaluated at the origin's absolute time indices with the frozen draws;
        for the SV error model the per-draw log-volatility is **filtered forward**
        through the realised residuals of the gap ``[n_train, origin)`` so the
        predictive intervals reflect the volatility at the origin rather than the
        (stale) value at the training end.  ``exog_history`` (aligned to
        ``y_history``) is required for that filtering whenever the generic block
        uses exogenous columns.
        """
        s = self.s
        S = s.num_draws
        if S == 0:
            raise RuntimeError("sampler has no retained draws; call run() first")
        H = int(horizon)
        mean, std = s.y_mean_, s.y_std_
        gen = s.np_rng

        y_hist = np.asarray(y_history, dtype=float).ravel()
        n_anchor = y_hist.shape[0]
        if n_anchor < self.n_full:
            raise ValueError("y_history shorter than the fitted training series")

        t_future = np.arange(n_anchor, n_anchor + H)
        trend_std = self._trend_future(t_future, time_future)        # (H, S)
        seasonal_std = self._seasonal_future(t_future, time_future)  # (H, S)

        noise_var = self._noise_variance_from_origin(
            y_hist, H, S, gen, exog_history, time_history, sv_filter_window)

        generic_std = np.zeros((H, S))
        paths = np.zeros((H, S))
        has_generic = s.generic_block is not None
        for si in range(S):
            hist = np.empty(n_anchor + H)
            hist[:n_anchor] = y_hist
            for h in range(H):
                t_raw = n_anchor + h
                g = 0.0
                if has_generic:
                    exog_row = self._exog_row(exog_future, h)
                    x_ge = self.b.future_generic_row(t_raw, hist, exog_row)
                    g = float(s.generic_block.predict_single(x_ge, si)[0])
                generic_std[h, si] = g
                eps = gen.normal(0.0, np.sqrt(noise_var[h, si]))
                z_tilde = trend_std[h, si] + seasonal_std[h, si] + g + eps
                y_tilde = mean + std * z_tilde
                hist[t_raw] = y_tilde
                paths[h, si] = y_tilde

        components = {
            "trend": mean + std * trend_std,
            "seasonal": std * seasonal_std,
            "generic": std * generic_std,
        }
        return ForecastResult(t_index=t_future, samples=paths,
                              components=components, backend=s.backend)

    # log chi^2_1 single-Gaussian approximation (mean, variance) for SV filtering
    _LOGCHI2_MEAN = -1.2703628454614782
    _LOGCHI2_VAR = 4.934802200544679

    def _noise_variance_from_origin(self, y_hist, H, S, gen, exog_history,
                                    time_history, window) -> np.ndarray:
        """Per-(step, draw) observation variance at the re-anchored origin."""
        s = self.s
        if not s.sv_mode:
            sig2 = np.asarray(s.sigma2_draws_, dtype=float)          # (S,)
            return np.tile(sig2[None, :], (H, 1))

        n_anchor = y_hist.shape[0]
        gap = n_anchor - self.n_full
        h_origin = np.asarray(s.h_last_draws_, dtype=float).copy()   # (S,)

        if gap > 0:
            W = int(min(gap, window))                  # only the recent gap matters
            t_gap = np.arange(n_anchor - W, n_anchor)
            z_gap = (y_hist[t_gap] - s.y_mean_) / s.y_std_           # (W,)
            trend_g = self._trend_future(t_gap, time_history)        # (W, S)
            seasonal_g = self._seasonal_future(t_gap, time_history)  # (W, S)
            if s.generic_block is not None:
                rows = [self.b.future_generic_row(
                            int(t), y_hist, self._realised_exog_row(exog_history, t))
                        for t in t_gap]
                generic_g = s.generic_block.predict_new(np.vstack(rows))  # (W, S)
            else:
                generic_g = np.zeros((W, S))
            resid = z_gap[:, None] - (trend_g + seasonal_g + generic_g)   # (W, S)
            h_origin = self._sv_filter_forward(h_origin, resid, gen)

        out = np.zeros((H, S))
        for si in range(S):
            out[:, si] = s.sv.forecast_path(h_origin[si], H, gen)
        return out

    def _sv_filter_forward(self, h_start, resid, gen) -> np.ndarray:
        """Forward Kalman filter of the AR(1) log-variance through realised
        residuals (single-Gaussian approx to log chi^2_1); returns a draw of the
        log-volatility at the origin for every retained sample.  Vectorised over
        draws; ``resid`` is ``(W, S)``."""
        sv = self.s.sv
        mu, phi, sh2 = sv.mu, sv.phi, sv.sigma_h2
        m_bar, v_bar = self._LOGCHI2_MEAN, self._LOGCHI2_VAR
        a = np.asarray(h_start, dtype=float).copy()      # (S,) filtered mean
        P = sh2 / (1.0 - phi ** 2)                        # scalar prior variance
        u = np.log(resid ** 2 + 1e-10) - m_bar           # (W, S) obs of h
        W = u.shape[0]
        for t in range(W):
            a_pred = mu + phi * (a - mu)
            P_pred = phi ** 2 * P + sh2
            Sinnov = P_pred + v_bar
            K = P_pred / Sinnov
            a = a_pred + K * (u[t] - a_pred)
            P = (1.0 - K) * P_pred
        return a + np.sqrt(max(P, 0.0)) * gen.standard_normal(a.shape[0])

    @staticmethod
    def _realised_exog_row(exog_history, t) -> Optional[Dict[str, float]]:
        if not exog_history:
            return None
        return {k: float(np.asarray(v)[t]) for k, v in exog_history.items()}
