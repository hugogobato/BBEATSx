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
