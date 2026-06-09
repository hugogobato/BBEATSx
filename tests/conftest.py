"""Shared fixtures / synthetic DGPs for the BBEATSx test suite.

By default the suite uses whichever forest backend auto-selects -- the production
``stochtree`` sampler when it is installed, otherwise the pure-numpy reference
backend. Force a choice with ``BBEATSX_BACKEND=stochtree`` or ``=numpy``.
"""

import numpy as np
import pytest


def linear_series(n=150, a=1.0, b=0.05, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = a + b * t + (rng.normal(0, noise, n) if noise > 0 else 0.0)
    return t, y


def seasonal_series(n=180, period=12, amp=2.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = amp * np.sin(2 * np.pi * t / period) + (rng.normal(0, noise, n)
                                                if noise > 0 else 0.0)
    return t, y


def full_series(n=200, period=12, trend_slope=0.04, amp=1.5, sigma=0.5, seed=0):
    """Trend + seasonal + AR(1) generic + homoscedastic noise, known parts."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 1.0 + trend_slope * t
    seasonal = amp * np.sin(2 * np.pi * t / period)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal(0, sigma)
    y = trend + seasonal + e
    return t, y, dict(trend=trend, seasonal=seasonal, noise=e)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
