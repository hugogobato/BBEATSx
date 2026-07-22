"""Analytic backend contract for heteroskedastic Gaussian forest leaves."""

from __future__ import annotations

import numpy as np
import pytest

from bbeatsx.backend import _numpy_backend


def _root_leaf_draws(backend, draws: int = 4000) -> np.ndarray:
    y = np.array([0.0, 0.0, 10.0, 10.0])
    variance = np.array([1.0, 1.0, 100.0, 100.0])
    dataset = backend.Dataset()
    dataset.add_covariates(np.arange(y.size, dtype=float)[:, None])
    dataset.add_variance_weights(variance)
    residual = backend.Residual(y.copy())
    global_config = backend.GlobalModelConfig(1.0)
    forest_config = backend.ForestModelConfig(
        num_trees=1,
        num_features=1,
        num_observations=y.size,
        feature_types=np.zeros(1, dtype=int),
        variable_weights=np.ones(1),
        alpha=0.95,
        beta=2.0,
        min_samples_leaf=1,
        max_depth=0,
        leaf_model_type=0,
        leaf_model_scale=1.0,
    )
    forest = backend.Forest(1, 1, True)
    container = backend.ForestContainer(1, 1, True)
    sampler = backend.ForestSampler(dataset, global_config, forest_config)
    sampler.prepare_for_sampler(dataset, residual, forest, 0, np.zeros(1))
    rng = backend.RNG(1729)
    for _ in range(draws):
        sampler.sample_one_iteration(
            container, forest, dataset, residual, rng, global_config,
            forest_config, True, False, 1,
        )
    return np.asarray(container.predict(dataset), dtype=float)[0]


def _assert_variance_multiplier_posterior(draws: np.ndarray) -> None:
    y = np.array([0.0, 0.0, 10.0, 10.0])
    precision = 1.0 / np.array([1.0, 1.0, 100.0, 100.0])
    expected_var = 1.0 / (1.0 + precision.sum())
    expected_mean = expected_var * np.dot(precision, y)
    assert draws.mean() == pytest.approx(expected_mean, abs=0.06)
    assert draws.var() == pytest.approx(expected_var, abs=0.04)


def test_numpy_backend_uses_variance_multipliers():
    _assert_variance_multiplier_posterior(_root_leaf_draws(_numpy_backend))


def test_stochtree_uses_the_same_observed_low_level_contract():
    stochtree = pytest.importorskip("stochtree")
    _assert_variance_multiplier_posterior(_root_leaf_draws(stochtree))
