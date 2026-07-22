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
    # F-2.4-1: the per-draw SV level must be stored with the retained draws.
    assert len(m.sampler_.mu_draws_) == 60
    assert np.std(m.sampler_.mu_draws_) > 0.0
    # F-2.4-8: the exact-target MH step should be on by default and accept often.
    assert m.sampler_.sv.exact
    assert 0.5 < m.sampler_.sv.mh_acceptance_rate <= 1.0
    fc = m.forecast(10)
    assert np.all(np.isfinite(fc.mean()))
    lo, hi = fc.interval(0.9)
    assert np.all(hi >= lo)


def test_sv_exact_flag_off_reproduces_prefix_sampler():
    """sv_exact=False must skip the MH step (pre-fix behaviour)."""
    _, y, _ = full_series(n=120, sigma=0.6, seed=8)
    cfg = make_config(periods=[(12, 2)], lags=(1,), trend="spline", errors="sv",
                      num_gfr=2, num_burnin=20, num_mcmc=30, seed=0)
    cfg.errors.sv_exact = False
    m = BBEATSx(cfg).fit(y)
    assert not m.sampler_.sv.exact
    assert m.sampler_.sv.mh_proposals == 0
    fc = m.forecast(5)
    assert np.all(np.isfinite(fc.mean()))
