"""BBEATSx -- Bayesian Basis Expansion Analysis for Time Series (exogenous).

An interpretable trend + seasonality + generic decomposition whose blocks are
Bayesian Additive Regression Tree ensembles (and an extrapolation-safe parametric
trend), fit by one coherent backfitting MCMC over ``stochtree`` low-level
primitives, delivering calibrated predictive intervals -- including per component.

See ``BBEATSx_research_plan.md`` (Phase 1) for the design this package implements.
"""

from __future__ import annotations

from .__about__ import __version__
from .backend import BACKEND
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
from .features import FeatureBuilder, FeatureSet
from .forecast import ForecastResult, Forecaster
from .model import BBEATSx, make_config
from .sampler import BBEATSxSampler

__all__ = [
    "__version__",
    "BACKEND",
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
]
