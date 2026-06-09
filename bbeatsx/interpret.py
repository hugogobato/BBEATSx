"""Interpretability instruments for BBEATSx (plan §1.4, §0.4).

All functions return numpy summaries (so they are testable and plotting-agnostic);
thin matplotlib helpers are provided for the common plots but import matplotlib
lazily so the core package has no hard plotting dependency.

Provided:

- :func:`decomposition` -- in-sample component posterior bands (the
  interpretability centerpiece of concept §2.3).
- :func:`split_importance` -- posterior of per-feature split frequencies for a
  forest block (BART §3 / BAVART; "which drivers matter, with uncertainty").
- :func:`partial_dependence` -- partial-dependence function with posterior
  intervals (BART eq. 19).
- :func:`girf` -- a generalized-impulse-response-style routine: intervene on an
  exogenous path and read the posterior of the response (BAVART template).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .blocks import ForestBlock
from .features import FeatureBuilder
from .forecast import Forecaster
from .sampler import BBEATSxSampler


@dataclass
class BandSummary:
    """Posterior mean and a central credible band over an axis of draws."""

    x: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_draws(cls, x, draws: np.ndarray, level: float = 0.9) -> "BandSummary":
        a = (1.0 - level) / 2.0
        return cls(
            x=np.asarray(x),
            mean=draws.mean(axis=1),
            lower=np.quantile(draws, a, axis=1),
            upper=np.quantile(draws, 1.0 - a, axis=1),
        )


def decomposition(sampler: BBEATSxSampler, level: float = 0.9) -> Dict[str, BandSummary]:
    """In-sample posterior bands for each component on the original data scale."""
    mean, std = sampler.y_mean_, sampler.y_std_
    comps = sampler.in_sample_components()
    t = sampler.fs.t_index
    out = {}
    # The standardization mean is reported as part of the trend (the level).
    out["trend"] = BandSummary.from_draws(t, mean + std * comps["trend"], level)
    out["seasonal"] = BandSummary.from_draws(t, std * comps["seasonal"], level)
    out["generic"] = BandSummary.from_draws(t, std * comps["generic"], level)
    total = std * (comps["trend"] + comps["seasonal"] + comps["generic"]) + mean
    out["fitted"] = BandSummary.from_draws(t, total, level)
    return out


def _forest_block(sampler: BBEATSxSampler, block: str) -> ForestBlock:
    b = {"trend": sampler.trend_block,
         "seasonal": sampler.seasonal_block,
         "generic": sampler.generic_block}.get(block)
    if not isinstance(b, ForestBlock):
        raise ValueError(
            f"block '{block}' is not a forest block (it is {type(b).__name__}); "
            "split importance / partial dependence require a forest block")
    return b


@dataclass
class ImportanceSummary:
    names: List[str]
    mean: np.ndarray          # posterior mean split count per feature
    lower: np.ndarray
    upper: np.ndarray
    inclusion_prob: np.ndarray  # posterior P(feature used at least once)


def split_importance(sampler: BBEATSxSampler, block: str = "generic",
                     level: float = 0.9) -> ImportanceSummary:
    """Posterior split-frequency variable importance for a forest block."""
    b = _forest_block(sampler, block)
    counts = b.split_counts_per_draw()        # (S, p)
    a = (1.0 - level) / 2.0
    return ImportanceSummary(
        names=list(b.feature_names),
        mean=counts.mean(axis=0),
        lower=np.quantile(counts, a, axis=0),
        upper=np.quantile(counts, 1.0 - a, axis=0),
        inclusion_prob=(counts > 0).mean(axis=0),
    )


def partial_dependence(sampler: BBEATSxSampler, feature: int, grid: np.ndarray,
                       block: str = "generic", level: float = 0.9) -> BandSummary:
    """Partial-dependence function of one feature with posterior intervals.

    For each grid value the chosen feature is fixed across all training rows, the
    block is predicted (averaging over rows), and the posterior over draws is
    summarised.  Returned on the original-scale deviation units (``* y_std``).
    """
    b = _forest_block(sampler, block)
    std = sampler.y_std_
    grid = np.asarray(grid, dtype=float)
    pd_draws = np.zeros((grid.shape[0], b.container.num_samples()))
    base = b.X.copy()
    for gi, v in enumerate(grid):
        Xmod = base.copy()
        Xmod[:, feature] = v
        preds = b.predict_new(Xmod)            # (n, S)
        pd_draws[gi] = std * preds.mean(axis=0)
    return BandSummary.from_draws(grid, pd_draws, level)


@dataclass
class GIRFResult:
    horizon: np.ndarray
    response_mean: np.ndarray
    response_lower: np.ndarray
    response_upper: np.ndarray
    response_samples: np.ndarray   # (H, S) difference of predictive paths


def girf(sampler: BBEATSxSampler, builder: FeatureBuilder, y_full: np.ndarray,
         baseline_exog: Dict[str, np.ndarray], shocked_exog: Dict[str, np.ndarray],
         horizon: int, level: float = 0.9, time_future=None) -> GIRFResult:
    """Generalized impulse response: response to an exogenous-path intervention.

    Runs the recursive forecaster twice -- with ``baseline_exog`` and with
    ``shocked_exog`` over the horizon -- using the *same* posterior draws, and
    returns the posterior distribution of the per-step difference (BAVART §0.4).
    """
    fc = Forecaster(sampler, builder, y_full)
    base = fc.forecast(horizon, exog_future=baseline_exog, time_future=time_future)
    shock = fc.forecast(horizon, exog_future=shocked_exog, time_future=time_future)
    diff = shock.samples - base.samples
    a = (1.0 - level) / 2.0
    return GIRFResult(
        horizon=base.t_index,
        response_mean=diff.mean(axis=1),
        response_lower=np.quantile(diff, a, axis=1),
        response_upper=np.quantile(diff, 1.0 - a, axis=1),
        response_samples=diff,
    )


# --------------------------------------------------------------- plotting (opt)
def plot_decomposition(sampler: BBEATSxSampler, level: float = 0.9, ax=None):
    """Plot the component decomposition with credible bands (needs matplotlib)."""
    import matplotlib.pyplot as plt  # lazy

    bands = decomposition(sampler, level)
    keys = ["fitted", "trend", "seasonal", "generic"]
    if ax is None:
        _, ax = plt.subplots(len(keys), 1, figsize=(9, 8), sharex=True)
    for a, k in zip(np.atleast_1d(ax), keys):
        bs = bands[k]
        a.plot(bs.x, bs.mean, lw=1.5, label=f"{k} (mean)")
        a.fill_between(bs.x, bs.lower, bs.upper, alpha=0.3)
        a.set_ylabel(k)
        a.legend(loc="upper left", fontsize=8)
    return ax
