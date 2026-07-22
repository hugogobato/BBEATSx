"""BBEATSx -- Bayesian Basis Expansion Analysis for Time Series (exogenous).

An interpretable additive decomposition whose correction blocks are Bayesian
Additive Regression Tree ensembles (with an extrapolation-safe parametric trend),
fit by one shared-residual sampler over ``stochtree`` low-level primitives. The
legacy generic block can optionally be routed into separate exogenous and
autoregressive forests.

See ``BBEATSx_research_plan.md`` (Phase 1) for the design this package implements.
"""

from __future__ import annotations

from .__about__ import __version__
from .backend import BACKEND, BACKEND_VERSION
from .config import (
    BBEATSxConfig,
    ErrorConfig,
    GenericConfig,
    MCMCConfig,
    SeasonalConfig,
    SeasonalPeriod,
    TreePrior,
    TrendConfig,
)
from .diagnostics import (
    capture_ratio_curve,
    empirical_capture_ratio,
    predicted_capture_ratio,
)
from .features import FeatureBuilder, FeatureSet
from .forecast import ForecastResult, Forecaster
from .model import BBEATSx, make_config
from .sampler import BBEATSxSampler

__all__ = [
    "__version__",
    "BACKEND",
    "BACKEND_VERSION",
    "BBEATSx",
    "make_config",
    "BBEATSxConfig",
    "TrendConfig",
    "SeasonalConfig",
    "SeasonalPeriod",
    "GenericConfig",
    "ErrorConfig",
    "MCMCConfig",
    "TreePrior",
    "FeatureBuilder",
    "FeatureSet",
    "BBEATSxSampler",
    "Forecaster",
    "ForecastResult",
    "predicted_capture_ratio",
    "capture_ratio_curve",
    "empirical_capture_ratio",
]
