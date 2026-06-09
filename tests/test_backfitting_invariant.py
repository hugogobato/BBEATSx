"""Backfitting residual invariant (plan §1.5 test b).

After every sweep the shared residual must satisfy ``z - sum_c F_c - r ~= 0``.
"""

import numpy as np

from bbeatsx import BBEATSx, make_config
from conftest import full_series


def test_residual_invariant_homoscedastic():
    _, y, _ = full_series(n=160, seed=4)
    cfg = make_config(periods=[(12, 3)], lags=(1, 2), trend="spline", errors="homo",
                      num_gfr=5, num_burnin=30, num_mcmc=60, seed=0)
    m = BBEATSx(cfg).fit(y)
    assert m.sampler_.backfitting_residual_error() < 1e-8


def test_residual_invariant_sv_and_tvp():
    _, y, _ = full_series(n=160, seed=5)
    cfg = make_config(periods=[(12, 2)], lags=(1,), trend="tvp", errors="sv",
                      num_gfr=3, num_burnin=30, num_mcmc=50, seed=0)
    m = BBEATSx(cfg).fit(y)
    assert m.sampler_.backfitting_residual_error() < 1e-7
