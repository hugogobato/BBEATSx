"""Identification of the additive level (``BBEATSxConfig.level_gauge``).

An additive decomposition pins down ``F_tr + F_se + F_ge`` but not how a constant
is shared among the blocks, so the *reported* components are only meaningful once
a gauge is fixed.  The default ``level_gauge="trend"`` folds each draw's seasonal
and generic in-sample means into the trend.  These tests assert the two
properties that make that safe: it is exactly predictive-invariant, and it really
does zero the two deviation blocks.
"""

import numpy as np

from bbeatsx import BBEATSx, make_config
from conftest import full_series


def _fit(y, gauge, seed=0):
    cfg = make_config(periods=[(12, 2)], lags=(1,), trend="spline", errors="homo",
                      level_gauge=gauge, num_gfr=8, num_burnin=100, num_mcmc=150,
                      seed=seed)
    return BBEATSx(cfg).fit(y)


def test_gauge_is_predictive_invariant():
    """Fixing the gauge must not move a single predictive draw."""
    t, y, _ = full_series(n=200, seed=3)
    free, fixed = _fit(y, "none", seed=3), _fit(y, "trend", seed=3)
    fc_free, fc_fixed = free.forecast(12), fixed.forecast(12)
    assert np.array_equal(fc_free.samples, fc_fixed.samples)
    # The component *sum* is invariant even though the split is not.
    tot_free = sum(fc_free.components[k] for k in ("trend", "seasonal", "generic"))
    tot_fixed = sum(fc_fixed.components[k] for k in ("trend", "seasonal", "generic"))
    assert np.allclose(tot_free, tot_fixed, atol=1e-10)


def test_gauge_zeroes_the_deviation_blocks():
    """Under the trend gauge, seasonal and generic are mean-zero in sample."""
    t, y, _ = full_series(n=200, seed=4)
    m = _fit(y, "trend", seed=4)
    comps = m.sampler_.in_sample_components()
    assert abs(comps["seasonal"].mean()) < 1e-10
    assert abs(comps["generic"].mean()) < 1e-10
    # ... and the trend then carries the level, i.e. the series mean.
    s = m.sampler_
    trend = s.y_mean_ + s.y_std_ * comps["trend"]
    assert abs(trend.mean() - s.y_mean_) < 0.1 * s.y_std_


def test_gauge_is_shared_by_fit_and_forecast():
    """The forecast decomposition must use the same (in-sample) gauge."""
    t, y, _ = full_series(n=200, seed=5)
    m = _fit(y, "trend", seed=5)
    fc = m.forecast(12)
    off = m.sampler_.level_offsets()
    # The offsets are per-draw constants that were actually moved.
    assert off["seasonal"].shape == (m.sampler_.num_draws,)
    assert np.any(off["seasonal"] != 0.0)
    # Components still reconstruct the predictive mean (noise averages out).
    total = sum(fc.components[k] for k in ("trend", "seasonal", "generic"))
    assert np.max(np.abs(total.mean(axis=1) - fc.mean())) < 0.5


def test_free_gauge_leaves_a_floating_level():
    """The old behaviour is preserved under ``level_gauge="none"`` (ablation)."""
    t, y, _ = full_series(n=200, seed=6)
    m = _fit(y, "none", seed=6)
    comps = m.sampler_.in_sample_components()
    off = m.sampler_.level_offsets()
    assert np.all(off["seasonal"] == 0.0) and np.all(off["generic"] == 0.0)
    # Nothing forces the deviation blocks to be centered here.
    assert abs(comps["seasonal"].mean()) + abs(comps["generic"].mean()) > 0.0
