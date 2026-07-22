"""High-level BBEATSx estimator and adapters (plan §1.5).

:class:`BBEATSx` is the user-facing object: an sklearn-style ``fit`` / ``forecast``
wrapper around the feature builder (§1.1), the Gibbs engine (§1.2), the recursive
forecaster (§1.3) and the interpretability tools (§1.4).  A thin Nixtla
(long-format) adapter lets BBEATSx drop into the Phase-0 comparison harness.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import backend as bk
from . import interpret as _interp
from .config import (
    BBEATSxConfig, ErrorConfig, GenericConfig, MCMCConfig, SeasonalConfig,
    SeasonalPeriod, TrendConfig,
)
from .features import FeatureBuilder
from .forecast import ForecastResult, Forecaster
from .sampler import BBEATSxSampler


def make_config(
    periods: Optional[Sequence] = None,
    lags: Sequence[int] = (1,),
    exog: Sequence[str] = (),
    future_exog: Sequence[str] = (),
    component_layout: str = "combined",
    calendar: Sequence[str] = (),
    trend: str = "spline",
    errors: str = "homo",
    asymmetric: bool = True,
    sum_to_zero: bool = True,
    level_gauge: str = "trend",
    deslope: bool = True,
    learn_rw_var: bool = False,
    num_gfr: int = 10,
    num_burnin: int = 200,
    num_mcmc: int = 500,
    thin: int = 1,
    seed: int = 0,
    num_threads: int = 1,
) -> BBEATSxConfig:
    """Convenience builder for a :class:`BBEATSxConfig`.

    ``periods`` accepts ``SeasonalPeriod`` instances or ``(period, harmonics)``
    tuples / bare period floats.
    """
    sp: List[SeasonalPeriod] = []
    for p in (periods or []):
        if isinstance(p, SeasonalPeriod):
            sp.append(p)
        elif isinstance(p, (tuple, list)):
            sp.append(SeasonalPeriod(float(p[0]), int(p[1]) if len(p) > 1 else 3))
        else:
            sp.append(SeasonalPeriod(float(p)))
    return BBEATSxConfig(
        trend=TrendConfig(mode=trend, deslope=deslope, learn_rw_var=learn_rw_var),
        seasonal=SeasonalConfig(periods=sp, calendar=list(calendar),
                                sum_to_zero=sum_to_zero),
        generic=GenericConfig(lags=tuple(lags), exog=list(exog),
                              future_exog=list(future_exog),
                              component_layout=component_layout,
                              asymmetric=asymmetric),
        errors=ErrorConfig(mode=errors),
        mcmc=MCMCConfig(num_gfr=num_gfr, num_burnin=num_burnin,
                        num_mcmc=num_mcmc, thin=thin, seed=seed,
                        num_threads=num_threads),
        level_gauge=level_gauge,
    )


class BBEATSx:
    """Bayesian Basis Expansion Analysis for Time Series (exogenous-capable)."""

    def __init__(self, config: Optional[BBEATSxConfig] = None, **kwargs) -> None:
        if config is None:
            config = make_config(**kwargs)
        elif kwargs:
            raise ValueError("pass either `config` or keyword args, not both")
        self.config = config
        self.builder_: Optional[FeatureBuilder] = None
        self.sampler_: Optional[BBEATSxSampler] = None
        self.forecaster_: Optional[Forecaster] = None
        self.y_full_: Optional[np.ndarray] = None
        self.backend = bk.BACKEND
        self.backend_version = bk.BACKEND_VERSION

    # ------------------------------------------------------------------- fit
    def fit(self, y: np.ndarray, time=None, exog=None) -> "BBEATSx":
        """Fit BBEATSx on a single series.

        Parameters
        ----------
        y : array-like
            The target series (chronological order).
        time : pandas.DatetimeIndex, optional
            Timestamps, required only if calendar seasonal features are used.
        exog : dict or DataFrame, optional
            Exogenous covariates aligned to ``y`` (length == len(y)).
        """
        y = np.asarray(y, dtype=float).ravel()
        self.y_full_ = y
        self.builder_ = FeatureBuilder(self.config.trend, self.config.seasonal,
                                       self.config.generic)
        fs = self.builder_.fit_transform(y, time=time, exog=exog)
        self.sampler_ = BBEATSxSampler(fs, self.config).run()
        self.forecaster_ = Forecaster(self.sampler_, self.builder_, y)
        return self

    def _check_fitted(self) -> None:
        if self.sampler_ is None:
            raise RuntimeError("model is not fitted; call fit() first")

    # -------------------------------------------------------------- forecast
    def forecast(self, horizon: int, exog_future: Optional[Dict] = None,
                 time_future=None) -> ForecastResult:
        self._check_fitted()
        return self.forecaster_.forecast(horizon, exog_future=exog_future,
                                         time_future=time_future)

    def forecast_from_origin(self, y_history, horizon: int,
                             exog_future: Optional[Dict] = None,
                             exog_history: Optional[Dict] = None,
                             time_future=None, time_history=None,
                             sv_filter_window: int = 256) -> ForecastResult:
        """Re-anchored forecast with frozen draws (no refit).

        Forecast ``horizon`` steps from the end of ``y_history`` using the
        parameters sampled at :meth:`fit` time. Enables periodic-recalibration
        protocols (fit once, then roll the forecast origin across realised data).
        See :meth:`Forecaster.forecast_from_origin`.
        """
        self._check_fitted()
        return self.forecaster_.forecast_from_origin(
            y_history, horizon, exog_future=exog_future,
            exog_history=exog_history, time_future=time_future,
            time_history=time_history, sv_filter_window=sv_filter_window)

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        """Posterior-mean point forecast (convenience over :meth:`forecast`)."""
        return self.forecast(horizon, **kwargs).mean()

    # --------------------------------------------------------- interpretation
    def decomposition(self, level: float = 0.9):
        self._check_fitted()
        return _interp.decomposition(self.sampler_, level)

    def split_importance(self, block: str = "generic", level: float = 0.9):
        self._check_fitted()
        return _interp.split_importance(self.sampler_, block, level)

    def partial_dependence(self, feature: int, grid, block: str = "generic",
                           level: float = 0.9):
        self._check_fitted()
        return _interp.partial_dependence(self.sampler_, feature, grid, block, level)

    def girf(self, baseline_exog, shocked_exog, horizon, level: float = 0.9,
             time_future=None):
        self._check_fitted()
        return _interp.girf(self.sampler_, self.builder_, self.y_full_,
                            baseline_exog, shocked_exog, horizon, level, time_future)

    # ------------------------------------------------------- Nixtla adapter
    def fit_nixtla(self, df, exog_cols: Optional[Sequence[str]] = None,
                   id_col: str = "unique_id", time_col: str = "ds",
                   target_col: str = "y", parse_time: bool = True) -> "BBEATSx":
        """Fit from a single-series Nixtla long-format DataFrame."""
        import pandas as pd

        d = df.sort_values(time_col)
        if d[id_col].nunique() > 1:
            raise ValueError("fit_nixtla handles one series; loop over unique_id "
                             "for panels")
        y = d[target_col].to_numpy(dtype=float)
        time = pd.DatetimeIndex(d[time_col]) if parse_time else None
        exog = ({c: d[c].to_numpy(dtype=float) for c in exog_cols}
                if exog_cols else None)
        self._nixtla_id_ = d[id_col].iloc[0]
        self._nixtla_last_ts_ = d[time_col].iloc[-1]
        self._nixtla_freq_ = (pd.infer_freq(pd.DatetimeIndex(d[time_col]))
                              if parse_time else None)
        return self.fit(y, time=time, exog=exog)

    def forecast_nixtla(self, horizon: int, levels: Sequence[int] = (80, 90),
                        exog_future: Optional[Dict] = None, time_future=None):
        """Forecast as a Nixtla-style DataFrame with mean and interval columns."""
        import pandas as pd

        res = self.forecast(horizon, exog_future=exog_future, time_future=time_future)
        out = {"unique_id": [getattr(self, "_nixtla_id_", 0)] * horizon}
        if getattr(self, "_nixtla_freq_", None):
            out["ds"] = pd.date_range(self._nixtla_last_ts_, periods=horizon + 1,
                                      freq=self._nixtla_freq_)[1:]
        else:
            out["ds"] = res.t_index
        out["BBEATSx"] = res.mean()
        for lv in levels:
            lo, hi = res.interval(lv / 100.0)
            out[f"BBEATSx-lo-{lv}"] = lo
            out[f"BBEATSx-hi-{lv}"] = hi
        return pd.DataFrame(out)
