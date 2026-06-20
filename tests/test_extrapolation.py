"""Extrapolation (in)consistency of the trend block (WP-2.3 / theory 23).

Pins the slope-leakage finding (Prop 2.3.6) and the de-sloping fix (F2):

  * the prior-predicted capture ratio matches the registered constant curve,
  * the de-sloping map is exactly in-sample-invariant,
  * the shipped spline extrapolates only ``rho ~ 0.83`` of a true slope while the
    de-sloped spline (and ``linear``) extrapolate the whole slope,
  * the ``tree`` foil's forecast trend is constant in the horizon (Thm 2.3.3).
"""

import numpy as np
import pytest

from bbeatsx import (
    BBEATSx,
    TrendConfig,
    capture_ratio_curve,
    empirical_capture_ratio,
    make_config,
    predicted_capture_ratio,
)
from conftest import linear_series


# --------------------------------------------------------------------------
# Prior-only capture ratio (the sharp, falsifiable constant; Prop 2.3.6)
# --------------------------------------------------------------------------
def test_predicted_capture_ratio_matches_registered_curve():
    # Registered prediction (defaults n_knots=10, coef_scale=10):
    #   rho = 0.816, 0.829, 0.895, 0.979 at smoothing = 0, 1, 10, 100.
    curve = capture_ratio_curve(TrendConfig(mode="spline"), [0.0, 1.0, 10.0, 100.0])
    np.testing.assert_allclose(curve, [0.8164, 0.8293, 0.8954, 0.9785], atol=2e-3)


def test_predicted_capture_ratio_is_n_invariant():
    tc = TrendConfig(mode="spline", smoothing=1.0)
    assert abs(predicted_capture_ratio(tc, 150) - predicted_capture_ratio(tc, 400)) < 1e-6


def test_linear_mode_captures_full_slope():
    # [1, t] is full rank -> the slope is likelihood-identified -> rho == 1.
    assert predicted_capture_ratio(TrendConfig(mode="linear", degree=1)) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The de-sloping map is in-sample-invariant (moves only along null(Phi))
# --------------------------------------------------------------------------
def _fit_trend_only(y, trend="spline", deslope=True, **kw):
    cfg = make_config(periods=[], lags=(), trend=trend, deslope=deslope,
                      errors="homo", num_gfr=6, num_burnin=120, num_mcmc=300,
                      seed=0, **kw)
    return BBEATSx(cfg).fit(y)


def test_deslope_leaves_in_sample_fit_invariant():
    _, y = linear_series(n=160, a=1.0, b=0.05, noise=0.05, seed=7)
    on = _fit_trend_only(y, deslope=True).sampler_.in_sample_components()["trend"]
    off = _fit_trend_only(y, deslope=False).sampler_.in_sample_components()["trend"]
    # identical RNG path; de-sloping only post-processes draws along null(Phi).
    np.testing.assert_allclose(on, off, atol=1e-8)


# --------------------------------------------------------------------------
# Empirical capture ratio tracks the prior prediction (verification hook 1)
# --------------------------------------------------------------------------
def test_shipped_spline_leaks_slope_desloped_recovers_it():
    b = 0.1
    _, y = linear_series(n=200, a=0.0, b=b, noise=1e-3, seed=3)  # high-SNR line

    rho_pred = predicted_capture_ratio(TrendConfig(mode="spline"), n=200)
    assert 0.80 < rho_pred < 0.86                                # ~0.829

    fc_off = _fit_trend_only(y, deslope=False).forecast(30)
    fc_on = _fit_trend_only(y, deslope=True).forecast(30)

    cr_off = empirical_capture_ratio(fc_off, b)
    cr_on = empirical_capture_ratio(fc_on, b)

    # shipped spline under-extrapolates by the prior-allocated amount...
    assert abs(cr_off - rho_pred) < 0.06
    # ...and the de-sloping fix restores the full slope.
    assert abs(cr_on - 1.0) < 0.06
    assert cr_on > cr_off + 0.10


def test_tvp_rw_var_hyperprior_learns_and_forecasts():
    # WP-2.3 flag 4: the rw innovation variance is learned under an inverse-gamma
    # hyperprior instead of being fixed to coef_scale^2*smoothing/n.
    from conftest import full_series
    _, y, _ = full_series(n=160, seed=4)
    cfg = make_config(periods=[(12, 2)], lags=(1,), trend="tvp", errors="homo",
                      learn_rw_var=True, num_gfr=4, num_burnin=40, num_mcmc=80, seed=0)
    m = BBEATSx(cfg).fit(y)
    tb = m.sampler_.trend_block
    assert tb.learn_rw_var and len(tb.rw_var_draws) == 80
    assert np.all(np.asarray(tb.rw_var_draws) > 0)           # proper IG draws
    assert np.std(tb.rw_var_draws) > 0                       # actually sampled, not fixed
    assert np.all(np.isfinite(m.forecast(12).mean()))


def test_tree_trend_forecast_is_constant_in_horizon():
    _, y = linear_series(n=150, a=0.0, b=0.1, noise=0.05, seed=5)
    with pytest.warns(UserWarning, match="extrapolation-failure foil"):
        m = _fit_trend_only(y, trend="tree")
    trend = m.forecast(40).component_mean("trend")
    # every future point shares the last training leaf: increments ~ 0 (Thm 2.3.3).
    assert np.max(np.abs(np.diff(trend))) < 1e-6
