"""All model variants are switchable from config (plan §1.5 exit criterion)."""

import itertools

import numpy as np
import pytest

from bbeatsx import BBEATSx, make_config
from conftest import full_series


@pytest.mark.parametrize("trend,errors", list(itertools.product(
    ["tree", "linear", "spline", "tvp"], ["homo", "sv"])))
def test_variant_runs(trend, errors):
    _, y, _ = full_series(n=140, seed=2)
    cfg = make_config(periods=[(12, 2)], lags=(1,), trend=trend, errors=errors,
                      num_gfr=3, num_burnin=20, num_mcmc=30, seed=0)
    m = BBEATSx(cfg).fit(y)
    assert m.sampler_.num_draws == 30
    fc = m.forecast(8)
    assert np.all(np.isfinite(fc.mean()))


@pytest.mark.parametrize("asymmetric", [True, False])
def test_asymmetric_prior_toggle(asymmetric):
    _, y, _ = full_series(n=140, seed=6)
    cfg = make_config(periods=[(12, 2)], lags=(1, 2), trend="spline", errors="homo",
                      asymmetric=asymmetric, num_gfr=3, num_burnin=20, num_mcmc=30,
                      seed=0)
    m = BBEATSx(cfg).fit(y)
    # asymmetric prior => fewer trees in the generic block
    gp = cfg.generic.resolved_tree_prior()
    assert m.sampler_.generic_block.tree_prior.num_trees == gp.num_trees
    if asymmetric:
        assert gp.num_trees < cfg.generic.tree_prior.num_trees


def test_sum_to_zero_toggle_changes_seasonal_centering():
    _, y, _ = full_series(n=144, seed=9)
    for s2z, expect_zero in [(True, True), (False, False)]:
        cfg = make_config(periods=[(12, 3)], lags=(1,), sum_to_zero=s2z,
                          num_gfr=2, num_burnin=10, num_mcmc=10, seed=0)
        m = BBEATSx(cfg).fit(y)
        means = m.builder_.seasonal_means_
        if expect_zero:
            assert means is not None
        else:
            assert means is None
