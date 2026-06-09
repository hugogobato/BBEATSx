"""Seasonal recovery via the forest block (plan §1.5 test a, seasonal half)."""

import numpy as np

from bbeatsx import BBEATSx, make_config
from conftest import seasonal_series


def test_recovers_sinusoid():
    t, y = seasonal_series(n=240, period=12, amp=2.0, noise=0.05, seed=1)
    cfg = make_config(periods=[(12, 3)], lags=(), trend="linear", errors="homo",
                      num_gfr=8, num_burnin=120, num_mcmc=200, seed=0)
    m = BBEATSx(cfg).fit(y)
    off = m.sampler_.fs.row_offset
    fitted = m.decomposition()["fitted"].mean
    truth = y[off:]
    rmse = np.sqrt(np.mean((fitted - truth) ** 2))
    corr = np.corrcoef(fitted, truth)[0, 1]
    assert corr > 0.9
    assert rmse < 0.5
