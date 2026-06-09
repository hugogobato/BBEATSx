"""Lightweight serialization for BBEATSx artifacts (plan §1.5).

Phase-1 scope: persist the *analysis artifacts* needed to reproduce a fitted
model's decomposition and calibration -- the config, the standardization
constants, the retained component draws and the error-variance draws -- to a
single ``.npz`` plus a JSON config sidecar.  This is enough to reload and inspect
a run, recompute component bands, and compare against baselines.

Full serialization of the fitted tree ensembles (so forecasting can resume in a
fresh process) is delegated to ``stochtree``'s ``JSONSerializer`` when running on
that backend; the numpy reference backend does not yet implement ensemble
round-tripping (the ensembles are cheap to refit).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict

import numpy as np

from .config import (
    BBEATSxConfig, ErrorConfig, GenericConfig, MCMCConfig, SeasonalConfig,
    SeasonalPeriod, TreePrior, TrendConfig,
)
from .sampler import BBEATSxSampler


def config_to_dict(config: BBEATSxConfig) -> Dict[str, Any]:
    """Recursively convert a config dataclass tree to a plain dict."""
    return dataclasses.asdict(config)


def config_from_dict(d: Dict[str, Any]) -> BBEATSxConfig:
    """Rebuild a :class:`BBEATSxConfig` from :func:`config_to_dict` output."""
    trend = TrendConfig(**{**d["trend"],
                           "tree_prior": TreePrior(**d["trend"]["tree_prior"])})
    seas = d["seasonal"]
    seasonal = SeasonalConfig(
        periods=[SeasonalPeriod(**p) for p in seas["periods"]],
        calendar=seas["calendar"], sum_to_zero=seas["sum_to_zero"],
        tree_prior=TreePrior(**seas["tree_prior"]),
    )
    gen = d["generic"]
    generic = GenericConfig(
        lags=tuple(gen["lags"]), exog=gen["exog"], future_exog=gen["future_exog"],
        asymmetric=gen["asymmetric"], tree_prior=TreePrior(**gen["tree_prior"]),
    )
    return BBEATSxConfig(
        trend=trend, seasonal=seasonal, generic=generic,
        errors=ErrorConfig(**d["errors"]), mcmc=MCMCConfig(**d["mcmc"]),
        multistep=d["multistep"],
    )


def save_run(sampler: BBEATSxSampler, path: str) -> None:
    """Save a fitted sampler's analysis artifacts to ``<path>.npz`` + ``.json``."""
    base = path[:-4] if path.endswith(".npz") else path
    comps = sampler.in_sample_components()
    arrays = {
        "trend": comps["trend"],
        "seasonal": comps["seasonal"],
        "generic": comps["generic"],
        "t_index": sampler.fs.t_index,
        "y_mean": np.array([sampler.y_mean_]),
        "y_std": np.array([sampler.y_std_]),
    }
    if sampler.sv_mode:
        arrays["sigma2_t"] = np.array(sampler.sigma2_t_draws_)
        arrays["h_last"] = np.array(sampler.h_last_draws_)
    else:
        arrays["sigma2"] = np.array(sampler.sigma2_draws_)
    np.savez_compressed(base + ".npz", **arrays)
    with open(base + ".json", "w") as fh:
        json.dump({"backend": sampler.backend,
                   "config": config_to_dict(sampler.config)}, fh, indent=2)


def load_run(path: str) -> Dict[str, Any]:
    """Load artifacts saved by :func:`save_run` into a dict."""
    base = path[:-4] if path.endswith(".npz") else path
    data = dict(np.load(base + ".npz"))
    with open(base + ".json") as fh:
        meta = json.load(fh)
    data["backend"] = meta["backend"]
    data["config"] = config_from_dict(meta["config"])
    return data
