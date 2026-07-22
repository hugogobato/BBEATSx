"""Static combined-versus-split generic component layouts."""

from __future__ import annotations

import numpy as np

from bbeatsx import BBEATSx, make_config
from bbeatsx.config import GenericConfig, SeasonalConfig, SeasonalPeriod, TrendConfig
from bbeatsx.features import FeatureBuilder
from bbeatsx.serialization import config_from_dict, config_to_dict


def _series(n=150, seed=91):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = rng.normal(size=n)
    state = np.zeros(n)
    for i in range(1, n):
        state[i] = 0.45 * state[i - 1] + rng.normal(0.0, 0.25)
    y = 1.0 + 0.02 * t + np.sin(2.0 * np.pi * t / 12.0) + 1.2 * x + state
    return y, {"x": x}


def test_feature_builder_exposes_legacy_and_split_views():
    y, exog = _series(n=80)
    builder = FeatureBuilder(
        TrendConfig(),
        SeasonalConfig(periods=[SeasonalPeriod(12, 2)]),
        GenericConfig(lags=(1, 2), exog=["x"], component_layout="split"),
    )
    fs = builder.fit_transform(y, exog=exog)
    assert fs.names_ar == ["y_lag1", "y_lag2"]
    assert fs.names_ex == ["x_x"]
    assert fs.names_ge == fs.names_ar + fs.names_ex
    assert np.array_equal(fs.X_ge, np.column_stack([fs.X_ar, fs.X_ex]))

    history = np.concatenate([y, [999.0]])
    assert np.array_equal(
        builder.future_autoregressive_row(80, history),
        builder.future_generic_row(80, history, {"x": 3.5})[:, :2],
    )
    assert builder.future_exogenous_row({"x": 3.5})[0, 0] == 3.5


def test_split_prior_preserves_tree_and_variance_budget():
    generic = GenericConfig(lags=(1,), exog=["x"], component_layout="split")
    legacy = generic.resolved_tree_prior()
    priors = generic.resolved_component_priors(
        has_exogenous=True, has_autoregressive=True)
    assert sum(p.num_trees for p in priors.values()) == legacy.num_trees
    legacy_var = legacy.num_trees * legacy.resolved_leaf_scale() ** 2
    split_var = sum(p.num_trees * p.resolved_leaf_scale() ** 2
                    for p in priors.values())
    assert split_var == legacy_var

    ar_only = generic.resolved_component_priors(
        has_exogenous=False, has_autoregressive=True)
    assert ar_only["autoregressive"] == legacy


def test_split_layout_returns_four_atomic_components_and_drawwise_gauge():
    y, exog = _series()
    n_train, horizon = 130, 12
    cfg = make_config(
        periods=[(12, 2)], lags=(1,), exog=["x"], component_layout="split",
        num_gfr=6, num_burnin=60, num_mcmc=100, seed=17,
    )
    model = BBEATSx(cfg).fit(y[:n_train], exog={"x": exog["x"][:n_train]})
    sampler = model.sampler_
    assert sampler.atomic_component_names == (
        "trend", "seasonal", "exogenous", "autoregressive")
    comps = sampler.in_sample_components()
    for name in ("seasonal", "exogenous", "autoregressive"):
        assert np.allclose(comps[name].mean(axis=0), 0.0, atol=1e-12)

    fc = model.forecast(horizon, exog_future={"x": exog["x"][n_train:]})
    assert tuple(fc.components) == sampler.atomic_component_names
    for name in sampler.atomic_component_names:
        assert fc.components[name].shape == (horizon, sampler.num_draws)


def test_ar_only_split_is_exactly_the_legacy_model_relabelled():
    y, _ = _series(n=130)
    common = dict(
        periods=[(12, 2)], lags=(1,), num_gfr=5, num_burnin=50,
        num_mcmc=80, seed=29,
    )
    combined = BBEATSx(make_config(**common, component_layout="combined")).fit(y)
    split = BBEATSx(make_config(**common, component_layout="split")).fit(y)
    fc_combined = combined.forecast(8)
    fc_split = split.forecast(8)
    assert np.array_equal(fc_combined.samples, fc_split.samples)
    assert np.array_equal(
        fc_combined.components["generic"],
        fc_split.components["autoregressive"],
    )


def test_component_layout_roundtrips_through_config_serialization():
    config = make_config(component_layout="split", exog=["x"], lags=(1, 2))
    restored = config_from_dict(config_to_dict(config))
    assert restored.generic.component_layout == "split"
