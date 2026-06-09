"""Stochastic-volatility sampler tests (plan §0.2, §1.2 step 4)."""

import numpy as np

from bbeatsx import BBEATSx, make_config
from bbeatsx.sv import SVSampler
from conftest import full_series


def test_sv_tracks_two_volatility_regimes():
    n = 300
    rng = np.random.default_rng(3)
    var = np.where(np.arange(n) < n // 2, 0.1, 2.0)
    eps = rng.normal(0, np.sqrt(var))

    sv = SVSampler(n, phi=0.97, sigma_h=0.2)
    gen = np.random.default_rng(0)   # SVSampler is a Python-side numpy sampler
    sigma2_t = None
    for _ in range(40):              # a few sweeps to let h adapt
        sigma2_t = sv.step(eps, gen)

    lo = sigma2_t[: n // 2].mean()
    hi = sigma2_t[n // 2:].mean()
    assert hi > lo
    assert hi / lo > 3.0          # clearly separates the two regimes


def test_sv_end_to_end_runs():
    _, y, _ = full_series(n=160, sigma=0.6, seed=8)
    cfg = make_config(periods=[(12, 2)], lags=(1,), trend="spline", errors="sv",
                      num_gfr=4, num_burnin=40, num_mcmc=60, seed=0)
    m = BBEATSx(cfg).fit(y)
    assert m.sampler_.num_draws == 60
    assert len(m.sampler_.sigma2_t_draws_) == 60
    fc = m.forecast(10)
    assert np.all(np.isfinite(fc.mean()))
    lo, hi = fc.interval(0.9)
    assert np.all(hi >= lo)
