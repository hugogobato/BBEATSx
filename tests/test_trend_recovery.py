"""Trend recovery + extrapolation (plan §1.5 test a; §3.6 / Lemma 2.3)."""

import numpy as np

from bbeatsx import BBEATSx, make_config
from conftest import linear_series


def _fit_linear_only(y, trend="linear", **kw):
    cfg = make_config(periods=[], lags=(), trend=trend, errors="homo",
                      num_gfr=5, num_burnin=60, num_mcmc=120, seed=0, **kw)
    return BBEATSx(cfg).fit(y)


def test_recovers_pure_linear_trend_in_low_noise_limit():
    # Near-noiseless line: the conjugate linear trend should reconstruct it.
    t, y = linear_series(n=120, a=2.0, b=0.08, noise=1e-3, seed=1)
    m = _fit_linear_only(y, trend="linear")
    fitted = m.decomposition()["fitted"].mean
    rmse = np.sqrt(np.mean((fitted - y[m.sampler_.fs.row_offset:]) ** 2))
    assert rmse < 0.05


def test_linear_trend_extrapolates_with_correct_slope():
    t, y = linear_series(n=120, a=0.0, b=0.1, noise=1e-3, seed=2)
    m = _fit_linear_only(y, trend="linear")
    fc = m.forecast(20)
    slope = np.median(np.diff(fc.mean()))
    assert abs(slope - 0.1) < 0.02


def test_tree_trend_flatlines_on_extrapolation():
    # The tree-on-t foil cannot continue a slope: its forecast trend component
    # is (near) constant, unlike the linear trend. This is the design-justifying
    # failure of plan §3.6 / Lemma 2.3.
    t, y = linear_series(n=150, a=0.0, b=0.1, noise=0.05, seed=3)
    m_tree = _fit_linear_only(y, trend="tree")
    fc_tree = m_tree.forecast(30)
    tree_trend = fc_tree.component_mean("trend")
    tree_spread = tree_trend.max() - tree_trend.min()

    m_lin = _fit_linear_only(y, trend="linear")
    lin_trend = m_lin.forecast(30).component_mean("trend")
    lin_spread = lin_trend.max() - lin_trend.min()

    # the linear trend keeps climbing over 30 steps (~0.1*30=3); the tree barely moves
    assert lin_spread > 2.0
    assert tree_spread < 0.5 * lin_spread
