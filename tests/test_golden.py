"""Reproducibility / golden-run regression test (plan §1.5 test d)."""

import numpy as np

from bbeatsx import BBEATSx, make_config
from bbeatsx import serialization
from conftest import full_series


def _fit():
    _, y, _ = full_series(n=150, seed=10)
    cfg = make_config(periods=[(12, 2)], lags=(1, 2), trend="spline", errors="homo",
                      num_gfr=4, num_burnin=40, num_mcmc=80, seed=42)
    return BBEATSx(cfg).fit(y), y


def test_fixed_seed_is_deterministic():
    m1, _ = _fit()
    m2, _ = _fit()
    f1 = m1.forecast(10).mean()
    f2 = m2.forecast(10).mean()
    assert np.allclose(f1, f2, atol=1e-12)
    d1 = m1.decomposition()["fitted"].mean
    d2 = m2.decomposition()["fitted"].mean
    assert np.allclose(d1, d2, atol=1e-12)


def test_serialization_roundtrip(tmp_path):
    m, _ = _fit()
    path = str(tmp_path / "run")
    serialization.save_run(m.sampler_, path)
    loaded = serialization.load_run(path)
    assert loaded["backend"] == m.backend
    assert loaded["trend"].shape == m.sampler_.in_sample_components()["trend"].shape
    assert loaded["config"].trend.mode == "spline"
    assert np.allclose(loaded["y_std"][0], m.sampler_.y_std_)
