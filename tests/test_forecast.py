"""Forecasting behaviour and calibration (plan §1.3, §1.5 exit criterion)."""

import numpy as np

from bbeatsx import BBEATSx, make_config
from conftest import full_series


def _fit(n=200, **kw):
    _, y, parts = full_series(n=n, seed=11)
    cfg = make_config(periods=[(12, 3)], lags=(1, 2), trend="spline", errors="homo",
                      num_gfr=6, num_burnin=80, num_mcmc=150, seed=0, **kw)
    return BBEATSx(cfg).fit(y), y, parts


def test_forecast_shapes_and_ordering():
    m, y, _ = _fit()
    fc = m.forecast(15)
    assert fc.samples.shape[1] == m.sampler_.num_draws
    assert fc.samples.shape[0] == 15
    lo, hi = fc.interval(0.9)
    assert np.all(hi >= lo)
    for k in ("trend", "seasonal", "generic"):
        assert fc.components[k].shape == (15, m.sampler_.num_draws)


def test_predictive_interval_widens_with_horizon():
    m, y, _ = _fit()
    fc = m.forecast(20)
    lo, hi = fc.interval(0.9)
    width = hi - lo
    # recursive roll-forward inflates variance: later horizons are wider on average
    assert width[-5:].mean() > width[:5].mean()


def test_in_sample_predictive_coverage_near_nominal():
    m, y, _ = _fit(n=200)
    s = m.sampler_
    comps = s.in_sample_components()
    total_std = comps["trend"] + comps["seasonal"] + comps["generic"]   # (n, S)
    sig = np.sqrt(np.array(s.sigma2_draws_))                            # (S,)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, total_std.shape) * sig[None, :]
    pred = s.y_mean_ + s.y_std_ * (total_std + noise)                  # (n, S)
    truth = y[s.fs.row_offset:]
    lo = np.quantile(pred, 0.05, axis=1)
    hi = np.quantile(pred, 0.95, axis=1)
    cov = np.mean((truth >= lo) & (truth <= hi))
    assert 0.80 <= cov <= 1.0
