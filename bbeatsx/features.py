"""Feature construction for BBEATSx (plan §1.1).

Builds three *disjoint* feature groups, one per additive block:

- **trend** ``X_tr``  : a basis ``phi(t)`` over a normalized time index
  (polynomial for ``linear``, an extrapolation-safe P-spline design for
  ``spline``).  These features are deterministic functions of ``t`` only and are
  never shared with the generic block.
- **seasonal** ``X_se`` : Fourier harmonics ``sin/cos(2*pi*k*t/p)`` plus optional
  calendar one-hots.  All known into the future, so the block extrapolates
  safely.
- **generic** ``X_ge`` : autoregressive lags ``y_{t-l}`` and exogenous covariates.

The :class:`FeatureBuilder` also knows how to assemble a *single future row* given
a rolled-forward history -- the primitive the recursive forecaster (§1.3) calls
draw-by-draw.

Disjointness (concept §3.5, plan §1.1) is enforced structurally: the trend and
seasonal column sources are physically distinct from the generic column sources,
and :meth:`FeatureBuilder.assert_disjoint` verifies no time-only feature leaked
into the generic block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # scipy is a hard dependency but guard the import for a clear message
    from scipy.interpolate import BSpline
except Exception as exc:  # pragma: no cover
    raise ImportError("bbeatsx.features requires scipy") from exc

from .config import GenericConfig, SeasonalConfig, TrendConfig

ExogLike = Union[Dict[str, np.ndarray], "PandasFrame", None]


@dataclass
class FeatureSet:
    """Container of the three disjoint training design matrices.

    Attributes
    ----------
    X_tr, X_se, X_ge : np.ndarray
        ``(n_eff, p_c)`` design matrices for trend / seasonal / generic blocks.
    y : np.ndarray
        Targets aligned to the design matrices (initial rows without full lag
        history are dropped).
    t_index : np.ndarray
        Raw integer time index of each retained row.
    names_tr, names_se, names_ge : list of str
        Column names per block (for interpretability / split-frequency reports).
    trend_penalty_cols : np.ndarray
        Boolean mask over ``X_tr`` columns marking spline columns subject to the
        2nd-difference (P-spline) smoothing prior.
    row_offset : int
        Number of leading rows dropped to satisfy the maximum AR lag.
    """

    X_tr: np.ndarray
    X_se: np.ndarray
    X_ge: np.ndarray
    y: np.ndarray
    t_index: np.ndarray
    names_tr: List[str]
    names_se: List[str]
    names_ge: List[str]
    trend_penalty_cols: np.ndarray
    row_offset: int


class FeatureBuilder:
    """Stateful builder that fits feature transforms on training data.

    Parameters
    ----------
    trend, seasonal, generic : config dataclasses
        Block configurations from :mod:`bbeatsx.config`.
    """

    _CALENDAR_FIELDS = {
        "month": lambda idx: idx.month,
        "dayofweek": lambda idx: idx.dayofweek,
        "hour": lambda idx: idx.hour,
        "quarter": lambda idx: idx.quarter,
        "dayofyear": lambda idx: idx.dayofyear,
        "week": lambda idx: idx.isocalendar().week.to_numpy(),
    }

    def __init__(
        self,
        trend: TrendConfig,
        seasonal: SeasonalConfig,
        generic: GenericConfig,
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.generic = generic

        # Fitted state (populated by fit_transform).
        self.t0_: float = 0.0
        self.t_scale_: float = 1.0
        self.knots_: Optional[np.ndarray] = None
        self.spline_degree_: int = trend.spline_degree
        self.calendar_levels_: Dict[str, np.ndarray] = {}
        self.seasonal_means_: Optional[np.ndarray] = None
        self.max_lag_: int = max(self.generic.lags) if self.generic.lags else 0
        self._fitted = False

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _as_exog_frame(exog: ExogLike, n: int) -> Dict[str, np.ndarray]:
        """Normalize ``exog`` (dict / DataFrame / None) to a dict of arrays."""
        if exog is None:
            return {}
        if hasattr(exog, "to_dict") and hasattr(exog, "columns"):  # pandas DataFrame
            return {str(c): np.asarray(exog[c].to_numpy(), dtype=float)
                    for c in exog.columns}
        out = {}
        for k, v in exog.items():
            out[str(k)] = np.asarray(v, dtype=float)
        for k, v in out.items():
            if v.shape[0] != n:
                raise ValueError(
                    f"exogenous column '{k}' has length {v.shape[0]}, expected {n}")
        return out

    # ------------------------------------------------------------- trend basis
    def _fit_trend_basis(self, t_norm_train: np.ndarray) -> None:
        """Set up the spline knot vector on the training time range."""
        if self.trend.mode == "spline":
            n_knots = self.trend.n_knots
            deg = self.trend.spline_degree
            lo, hi = float(t_norm_train.min()), float(t_norm_train.max())
            inner = np.linspace(lo, hi, n_knots)
            # Clamped (repeated) boundary knots so the basis is well-defined.
            self.knots_ = np.concatenate(
                [np.repeat(inner[0], deg), inner, np.repeat(inner[-1], deg)]
            )

    def _trend_design(self, t_norm: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Return ``(X_tr, names, penalty_mask)`` for the given normalized times.

        For ``spline`` mode the spline columns are evaluated on ``t`` *clamped to
        the training range*, while an explicit linear term ``[1, t]`` carries any
        extrapolating slope -- giving linear (not cubic) extrapolation beyond the
        observed range, which is the extrapolation-safe behaviour of plan §3.6 /
        Lemma 2.3.
        """
        t_norm = np.asarray(t_norm, dtype=float).ravel()
        n = t_norm.shape[0]

        if self.trend.mode == "linear":
            cols = [np.ones(n)]
            names = ["trend_1"]
            for d in range(1, self.trend.degree + 1):
                cols.append(t_norm ** d)
                names.append(f"trend_t^{d}")
            X = np.column_stack(cols)
            penalty = np.zeros(X.shape[1], dtype=bool)
            return X, names, penalty

        if self.trend.mode == "spline":
            assert self.knots_ is not None
            lin = np.column_stack([np.ones(n), t_norm])
            lin_names = ["trend_1", "trend_t"]
            lo, hi = float(self.knots_[0]), float(self.knots_[-1])
            t_clamped = np.clip(t_norm, lo, hi)
            deg = self.spline_degree_
            n_basis = len(self.knots_) - deg - 1
            spl_cols = np.empty((n, n_basis))
            for j in range(n_basis):
                c = np.zeros(n_basis)
                c[j] = 1.0
                spl_cols[:, j] = BSpline(self.knots_, c, deg, extrapolate=False)(t_clamped)
            spl_cols = np.nan_to_num(spl_cols, nan=0.0)
            # Drop the first spline column to avoid collinearity with the intercept.
            spl_cols = spl_cols[:, 1:]
            spl_names = [f"trend_spl_{j}" for j in range(spl_cols.shape[1])]
            X = np.column_stack([lin, spl_cols])
            names = lin_names + spl_names
            penalty = np.array([False, False] + [True] * spl_cols.shape[1])
            return X, names, penalty

        if self.trend.mode == "tvp":
            # TVP trend stays linear-in-phi(t); amplitudes vary via the block.
            X = np.column_stack([np.ones(n), t_norm])
            names = ["trend_1", "trend_t"]
            penalty = np.zeros(X.shape[1], dtype=bool)
            return X, names, penalty

        # tree mode feeds engineered t-features to a forest (the foil).
        cols = [t_norm]
        names = ["trend_t"]
        # A couple of low-order powers give the tree more to split on.
        cols.append(t_norm ** 2)
        names.append("trend_t^2")
        X = np.column_stack(cols)
        penalty = np.zeros(X.shape[1], dtype=bool)
        return X, names, penalty

    # ---------------------------------------------------------- seasonal block
    def _fit_calendar(self, time: Optional["DatetimeIndexLike"]) -> None:
        self.calendar_levels_ = {}
        if not self.seasonal.calendar:
            return
        if time is None or not hasattr(time, "month"):
            raise ValueError(
                "Seasonal calendar features requested but no pandas DatetimeIndex "
                "was supplied as `time`.")
        for field in self.seasonal.calendar:
            if field not in self._CALENDAR_FIELDS:
                raise ValueError(f"Unsupported calendar field '{field}'")
            vals = np.asarray(self._CALENDAR_FIELDS[field](time))
            self.calendar_levels_[field] = np.unique(vals)

    def _seasonal_design(
        self, t_raw: np.ndarray, time: Optional["DatetimeIndexLike"]
    ) -> Tuple[np.ndarray, List[str]]:
        t_raw = np.asarray(t_raw, dtype=float).ravel()
        cols: List[np.ndarray] = []
        names: List[str] = []
        for sp in self.seasonal.periods:
            for k in range(1, sp.harmonics + 1):
                ang = 2.0 * np.pi * k * t_raw / sp.period
                cols.append(np.sin(ang))
                names.append(f"sin_p{sp.period:g}_k{k}")
                cols.append(np.cos(ang))
                names.append(f"cos_p{sp.period:g}_k{k}")
        if self.calendar_levels_:
            if time is None or not hasattr(time, "month"):
                raise ValueError("calendar features require a DatetimeIndex `time`")
            for field, levels in self.calendar_levels_.items():
                vals = np.asarray(self._CALENDAR_FIELDS[field](time))
                # Drop the last level for identifiability (one-hot with reference).
                for lev in levels[:-1]:
                    cols.append((vals == lev).astype(float))
                    names.append(f"cal_{field}_{lev}")
        if not cols:
            # No seasonal features configured -> empty design with zero columns.
            return np.zeros((t_raw.shape[0], 0)), []
        return np.column_stack(cols), names

    # ----------------------------------------------------------- generic block
    def _generic_design(
        self,
        y_full: np.ndarray,
        exog: Dict[str, np.ndarray],
        start: int,
        stop: int,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build generic features for rows ``start:stop`` of the full series.

        Lags index into ``y_full`` (which must contain enough history before
        ``start``); exogenous columns are sliced ``start:stop``.
        """
        cols: List[np.ndarray] = []
        names: List[str] = []
        idx = np.arange(start, stop)
        for lag in self.generic.lags:
            cols.append(y_full[idx - lag])
            names.append(f"y_lag{lag}")
        for name in list(self.generic.exog) + list(self.generic.future_exog):
            if name not in exog:
                raise ValueError(f"generic exogenous column '{name}' not provided")
            cols.append(exog[name][idx])
            names.append(f"x_{name}")
        if not cols:
            return np.zeros((stop - start, 0)), []
        return np.column_stack(cols), names

    # ------------------------------------------------------------------ public
    def fit_transform(
        self,
        y: np.ndarray,
        time: Optional["DatetimeIndexLike"] = None,
        exog: ExogLike = None,
    ) -> FeatureSet:
        """Fit transforms on training data and return the three design matrices."""
        y = np.asarray(y, dtype=float).ravel()
        n = y.shape[0]
        exog_d = self._as_exog_frame(exog, n)

        t_raw_full = np.arange(n, dtype=float)
        self.t0_ = 0.0
        self.t_scale_ = float(n - 1) if n > 1 else 1.0
        t_norm_full = (t_raw_full - self.t0_) / self.t_scale_

        self._fit_trend_basis(t_norm_full[self.max_lag_:])
        self._fit_calendar(time[self.max_lag_:] if time is not None else None)

        start = self.max_lag_
        stop = n
        t_raw = t_raw_full[start:stop]
        t_norm = t_norm_full[start:stop]
        time_slice = time[start:stop] if time is not None else None

        X_tr, names_tr, penalty = self._trend_design(t_norm)
        X_se, names_se = self._seasonal_design(t_raw, time_slice)
        X_ge, names_ge = self._generic_design(y, exog_d, start, stop)

        # Sum-to-zero centering of the seasonal block (concept §3.5): subtract
        # in-sample column means so the seasonal component carries no level (the
        # trend intercept absorbs it). The same shift is applied to future rows.
        if self.seasonal.sum_to_zero and X_se.shape[1] > 0:
            self.seasonal_means_ = X_se.mean(axis=0)
            X_se = X_se - self.seasonal_means_
        else:
            self.seasonal_means_ = None

        fs = FeatureSet(
            X_tr=X_tr, X_se=X_se, X_ge=X_ge,
            y=y[start:stop], t_index=t_raw.astype(int),
            names_tr=names_tr, names_se=names_se, names_ge=names_ge,
            trend_penalty_cols=penalty, row_offset=start,
        )
        self._fitted = True
        self.assert_disjoint(fs)
        return fs

    def build_future_row(
        self,
        t_raw: int,
        y_history: np.ndarray,
        time_point: Optional["TimestampLike"] = None,
        exog_future: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assemble one future row ``(x_tr, x_se, x_ge)``.

        Parameters
        ----------
        t_raw : int
            Absolute integer time index of the future point (continues past the
            training range, so the trend basis is evaluated out-of-sample).
        y_history : np.ndarray
            Series values up to ``t_raw - 1`` (training + already-simulated
            future), long enough to supply every AR lag.
        time_point : pandas Timestamp, optional
            Needed only if calendar seasonal features are configured.
        exog_future : dict, optional
            Future values of any exogenous covariates required by the blocks.
        """
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.fit_transform must be called first")
        exog_future = exog_future or {}

        t_norm = (float(t_raw) - self.t0_) / self.t_scale_
        x_tr, _, _ = self._trend_design(np.array([t_norm]))

        # Seasonal: wrap a length-1 DatetimeIndex if calendar features are used.
        time_idx = None
        if self.calendar_levels_:
            if time_point is None:
                raise ValueError("future calendar features require `time_point`")
            import pandas as pd  # local import; only needed with calendar features
            time_idx = pd.DatetimeIndex([time_point])
        x_se, _ = self._seasonal_design(np.array([float(t_raw)]), time_idx)
        if self.seasonal_means_ is not None and x_se.shape[1] > 0:
            x_se = x_se - self.seasonal_means_

        cols: List[float] = []
        for lag in self.generic.lags:
            cols.append(float(y_history[t_raw - lag]))
        for name in list(self.generic.exog) + list(self.generic.future_exog):
            if name not in exog_future:
                raise ValueError(f"future value for exogenous '{name}' not provided")
            cols.append(float(exog_future[name]))
        x_ge = np.array(cols, dtype=float).reshape(1, -1) if cols else np.zeros((1, 0))

        return x_tr, x_se, x_ge

    # ------------------------------------------------------- future designs
    def future_trend_design(self, t_raw_array: np.ndarray) -> np.ndarray:
        """Trend basis for an array of future (absolute) integer indices."""
        t_norm = (np.asarray(t_raw_array, dtype=float) - self.t0_) / self.t_scale_
        X, _, _ = self._trend_design(t_norm)
        return X

    def future_seasonal_design(
        self, t_raw_array: np.ndarray, time_index: Optional["DatetimeIndexLike"] = None
    ) -> np.ndarray:
        """Seasonal design for future indices (centered consistently with training)."""
        X, _ = self._seasonal_design(np.asarray(t_raw_array, dtype=float), time_index)
        if self.seasonal_means_ is not None and X.shape[1] > 0:
            X = X - self.seasonal_means_
        return X

    def future_generic_row(
        self, t_raw: int, y_history: np.ndarray,
        exog_future: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Generic features for one future row given a rolled-forward history."""
        exog_future = exog_future or {}
        cols: List[float] = []
        for lag in self.generic.lags:
            cols.append(float(y_history[t_raw - lag]))
        for name in list(self.generic.exog) + list(self.generic.future_exog):
            if name not in exog_future:
                raise ValueError(f"future value for exogenous '{name}' not provided")
            cols.append(float(exog_future[name]))
        return (np.array(cols, dtype=float).reshape(1, -1) if cols
                else np.zeros((1, 0)))

    def assert_disjoint(self, fs: FeatureSet) -> None:
        """Guarantee no trend/seasonal time feature leaked into the generic block.

        The generic block may only contain ``y_lag*`` and ``x_*`` columns; any
        ``trend_*``/``sin_``/``cos_``/``cal_`` name appearing there is a bug.
        """
        forbidden_prefixes = ("trend_", "sin_", "cos_", "cal_")
        bad = [nm for nm in fs.names_ge if nm.startswith(forbidden_prefixes)]
        if bad:
            raise AssertionError(
                f"disjointness violated: time-only features in generic block: {bad}")
        # Generic columns must be lags or exogenous.
        ok = all(nm.startswith("y_lag") or nm.startswith("x_") for nm in fs.names_ge)
        if not ok:
            raise AssertionError(
                f"unexpected generic feature names: {fs.names_ge}")
