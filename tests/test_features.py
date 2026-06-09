"""Feature construction tests (plan §1.1)."""

import numpy as np

from bbeatsx.config import GenericConfig, SeasonalConfig, SeasonalPeriod, TrendConfig
from bbeatsx.features import FeatureBuilder
from conftest import full_series


def _builder(trend_mode="spline", periods=((12, 3),), lags=(1, 2), sum_to_zero=True):
    trend = TrendConfig(mode=trend_mode)
    seasonal = SeasonalConfig(
        periods=[SeasonalPeriod(p, h) for p, h in periods], sum_to_zero=sum_to_zero)
    generic = GenericConfig(lags=lags)
    return FeatureBuilder(trend, seasonal, generic)


def test_disjointness_enforced():
    _, y, _ = full_series(n=120)
    fs = _builder().fit_transform(y)
    # generic block must contain only lag / exogenous columns
    assert all(nm.startswith("y_lag") or nm.startswith("x_") for nm in fs.names_ge)
    # no trend/seasonal feature leaked into the generic block
    assert not any(nm.startswith(("trend_", "sin_", "cos_", "cal_"))
                   for nm in fs.names_ge)


def test_shapes_and_offset():
    _, y, _ = full_series(n=120)
    fs = _builder(lags=(1, 2, 3)).fit_transform(y)
    assert fs.row_offset == 3
    n_eff = 120 - 3
    assert fs.X_tr.shape[0] == n_eff
    assert fs.X_se.shape[0] == n_eff
    assert fs.X_ge.shape[0] == n_eff
    assert fs.y.shape[0] == n_eff
    # one lag column per lag
    assert fs.X_ge.shape[1] == 3


def test_seasonal_sum_to_zero_centers_columns():
    _, y, _ = full_series(n=144)
    fs = _builder(sum_to_zero=True).fit_transform(y)
    assert np.allclose(fs.X_se.mean(axis=0), 0.0, atol=1e-10)


def test_spline_trend_extrapolates_linearly():
    # A pure line -> the spline design's future rows should keep increasing
    # (linear extrapolation), not flatline.
    n = 100
    y = 0.5 + 0.1 * np.arange(n)
    b = _builder(trend_mode="spline", periods=(), lags=(1,))
    b.fit_transform(y)
    fut = b.future_trend_design(np.arange(n, n + 10))
    # the linear column (index 1) keeps growing beyond the training range
    assert np.all(np.diff(fut[:, 1]) > 0)


def test_future_generic_row_uses_rolled_history():
    _, y, _ = full_series(n=80)
    b = _builder(periods=(), lags=(1, 2))
    b.fit_transform(y)
    hist = np.concatenate([y, [999.0]])
    row = b.future_generic_row(80, hist)  # t=80 -> lags index 79, 78
    assert row.shape == (1, 2)
    assert row[0, 0] == y[79]
    assert row[0, 1] == y[78]
