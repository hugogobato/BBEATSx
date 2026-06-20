"""Extrapolation diagnostics for the trend block (WP-2.3 verification hooks).

The headline is :func:`predicted_capture_ratio` -- the fraction ``rho`` of a true
linear-trend slope that the **shipped, un-de-sloped** spline trend actually
extrapolates (Prop 2.3.6 in ``theory/23_extrapolation.md``).  The shipped design
``[1, t, B_2..B_J]`` is *exactly* rank-deficient (clamped B-splines reproduce
linear functions), so the in-sample slope is split between the explicit ``beta_1``
channel -- the only one that extrapolates -- and the spline columns, which freeze
at the boundary.  The split is fixed by the prior precision alone, giving

    rho = 1 - v_1^2 / (coef_scale^2 * v' P v),

where ``v`` is the (1-d) null vector of the design and ``P`` the coefficient prior
precision.  ``rho`` is independent of ``n`` and of the data -- a constant computed
from the prior.  Matching an *empirical* capture ratio (extrapolated slope / true
slope, from actual fits) against this constant is therefore a sharp, falsifiable
check of the slope-leakage finding (verification hook 1):

  * ``tree`` trend       -> empirical capture ratio == 0 (flat; Thm 2.3.3),
  * shipped ``spline``   -> == ``predicted_capture_ratio`` (~0.83 under defaults),
  * de-sloped / ``linear`` -> == 1 (Thm 2.3.7).

Use :func:`empirical_capture_ratio` on a :class:`~bbeatsx.forecast.ForecastResult`
to get the left-hand side.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import GenericConfig, SeasonalConfig, TrendConfig
from .features import FeatureBuilder


def _trend_design_and_prec(trend: TrendConfig, n: int):
    """Build the shipped in-sample trend design and its prior precision.

    Reuses :class:`FeatureBuilder` so the knot vector, clamping and dropped
    column exactly match what the sampler sees (no re-derivation of the basis).
    """
    builder = FeatureBuilder(trend, SeasonalConfig(), GenericConfig(lags=()))
    fs = builder.fit_transform(np.zeros(int(n)))
    Phi = np.asarray(fs.X_tr, dtype=float)
    penalty = np.asarray(fs.trend_penalty_cols, dtype=bool)
    p = Phi.shape[1]

    P = np.eye(p) / (trend.coef_scale ** 2)
    idx = np.where(penalty)[0]
    if idx.size >= 3 and trend.smoothing > 0:
        k = idx.size
        D = np.zeros((k - 2, k))
        for i in range(k - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        full = np.zeros((p, p))
        full[np.ix_(idx, idx)] = D.T @ D
        P = P + trend.smoothing * full
    return Phi, P


def predicted_capture_ratio(trend: TrendConfig, n: int = 200) -> float:
    """Prior-predicted slope capture ratio ``rho`` of the *shipped* spline trend.

    Returns ``1.0`` for any full-rank design (``linear`` mode, or a spline design
    that happens to be full rank), since then the extrapolation slope is
    likelihood-identified.  Independent of ``n`` for the spline (the value is a
    prior constant); ``n`` only needs to exceed the basis dimension.
    """
    Phi, P = _trend_design_and_prec(trend, n)
    p = Phi.shape[1]
    if p < 2:
        return 1.0
    _, sv, Vt = np.linalg.svd(Phi, full_matrices=False)
    tol = sv[0] * max(Phi.shape) * np.finfo(float).eps
    if sv[-1] > tol:                       # full rank: slope identified by data
        return 1.0
    v = Vt[-1]
    if abs(v[1]) < 1e-12:
        return 1.0
    return float(1.0 - v[1] ** 2 / (trend.coef_scale ** 2 * float(v @ P @ v)))


def capture_ratio_curve(
    trend: TrendConfig, smoothings: Sequence[float], n: int = 200
) -> np.ndarray:
    """``predicted_capture_ratio`` over a grid of ``smoothing`` values.

    The registered prediction (defaults, ``n_knots=10``, ``coef_scale=10``) is
    ``rho = 0.816, 0.829, 0.895, 0.979`` at ``smoothing = 0, 1, 10, 100``.
    """
    out = np.empty(len(smoothings))
    for i, sm in enumerate(smoothings):
        tc = TrendConfig(
            mode="spline", n_knots=trend.n_knots, spline_degree=trend.spline_degree,
            coef_scale=trend.coef_scale, smoothing=float(sm),
        )
        out[i] = predicted_capture_ratio(tc, n)
    return out


def empirical_capture_ratio(forecast_result, true_slope: float) -> float:
    """Extrapolated trend slope / true per-step slope, from a forecast.

    ``forecast_result`` is a :class:`~bbeatsx.forecast.ForecastResult`; the trend
    slope is the median across draws of the per-step difference of the posterior
    trend component.  Compare against :func:`predicted_capture_ratio`.
    """
    trend = forecast_result.component_mean("trend")
    if trend.shape[0] < 2 or true_slope == 0.0:
        return float("nan")
    return float(np.median(np.diff(trend)) / true_slope)
