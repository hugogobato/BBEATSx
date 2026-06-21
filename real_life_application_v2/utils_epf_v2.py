"""Shared helpers for the BBEATSx real-life EPF study -- **v2 protocol**.

v2 = **periodic recalibration**.  Instead of refitting at every day-ahead origin
(v1, ``real_life_application/utils_epf.py``), each model is refit only every
``RECAL_BLOCK_DAYS`` (~6 months -> ~4 trainings over the ~728-day test set) on a
**fixed rolling window** (constant length, oldest days dropped at each recal).
Within a block the model is *not* refit: every day-ahead origin is forecast with
the **frozen** parameters, *re-anchored* to that origin by conditioning on the
realised series observed up to it (see ``bbeatsx.BBEATSx.forecast_from_origin``;
BART/neural/foundation models re-anchor analogously via their own predict paths).

This evaluates the full test set at ~4 fits per (model, market) instead of ~728,
while staying far fresher than a single training.  The loaders, metric battery and
sample reconstruction are reused verbatim from the v1 module so the two protocols
remain directly comparable.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Shared with v1 (loaders, metrics, sample reconstruction). The v2 notebooks put
# ``real_life_application`` on sys.path so this import resolves on Colab too.
from utils_epf import (  # noqa: F401
    load_market, load_split, make_nixtla_frames, samples_from_nf,
    samples_from_quantiles, evaluate_samples, MARKET_TEST_DATES,
    EXOG_COLS, HORIZON, FREQ, NF_LEVELS,
)

# ~6 months of daily origins; a ~728-day test set splits into 4 blocks.
RECAL_BLOCK_DAYS: int = 182


# --------------------------------------------------------------------------- #
# Recalibration schedule                                                      #
# --------------------------------------------------------------------------- #
def recal_blocks(n_test_days: int, block_days: int = RECAL_BLOCK_DAYS) -> List[tuple]:
    """Partition ``[0, n_test_days)`` into consecutive recalibration blocks.

    Each block ``(start_day, end_day)`` is refit once at ``start_day`` and then
    re-anchored across ``start_day .. end_day-1``.
    """
    blocks, s = [], 0
    while s < n_test_days:
        e = min(s + block_days, n_test_days)
        blocks.append((s, e))
        s = e
    return blocks


def default_train_window(test_start_pos: int) -> int:
    """Fixed rolling-window length (hours) = the market's *original* initial
    training span (data start -> test start), held constant and rolled forward."""
    return int(test_start_pos)


# --------------------------------------------------------------------------- #
# Per-block training slice (fixed rolling window)                             #
# --------------------------------------------------------------------------- #
def block_train_data(data: pd.DataFrame, test_start_pos: int, block_start_day: int,
                     train_window: int, horizon: int = HORIZON,
                     exog_cols: Optional[List[str]] = None) -> Dict:
    """Fixed rolling-window training slice for the block whose first origin is
    test day ``block_start_day``.

    Returns the training arrays plus the absolute indices needed to re-anchor the
    later origins of the block.  ``window_start_abs`` is the (fixed-length) window
    start; aligning every re-anchored history to this index keeps the trend basis
    consistent with the fitted model.
    """
    exog_cols = exog_cols or EXOG_COLS
    block_origin = test_start_pos + block_start_day * horizon
    window_start = max(0, block_origin - train_window)
    train = data.iloc[window_start:block_origin]
    return dict(
        window_start_abs=window_start,
        block_origin_abs=block_origin,
        y_train=train["Price"].to_numpy(float),
        exog_train={c: train[c].to_numpy(float) for c in exog_cols},
        ds_train=train.index,
    )


# --------------------------------------------------------------------------- #
# Per-origin re-anchoring slice (within a block, no refit)                    #
# --------------------------------------------------------------------------- #
def origin_eval_data(data: pd.DataFrame, test_start_pos: int, window_start_abs: int,
                     day: int, horizon: int = HORIZON,
                     exog_cols: Optional[List[str]] = None) -> Dict:
    """Re-anchoring slice for test ``day`` within a block fitted from
    ``window_start_abs``.

    ``y_history`` / ``exog_history`` run from the (fixed) window start up to the
    forecast origin -- i.e. the training window plus the realised gap since the
    block's refit -- so frozen-draw models can condition on the latest data
    without refitting.  ``exog_future`` is the known day-ahead covariate block.
    """
    exog_cols = exog_cols or EXOG_COLS
    origin = test_start_pos + day * horizon
    hist = data.iloc[window_start_abs:origin]
    fut = data.iloc[origin:origin + horizon]
    return dict(
        origin_abs=origin,
        y_history=hist["Price"].to_numpy(float),
        exog_history={c: hist[c].to_numpy(float) for c in exog_cols},
        exog_future={c: fut[c].to_numpy(float) for c in exog_cols},
        y_test=fut["Price"].to_numpy(float),
        ds_history=hist.index,
        ds_fut=fut.index,
    )


# --------------------------------------------------------------------------- #
# Nixtla long-format frames for the neural benchmarks                          #
# --------------------------------------------------------------------------- #
def block_nixtla_frame(bt: Dict, exog_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Long-format training frame for ``neuralforecast`` from a ``block_train_data``
    dict.  Used to fit a neural model **once per block**."""
    exog_cols = exog_cols or EXOG_COLS
    df = pd.DataFrame({"unique_id": "series_1", "ds": bt["ds_train"],
                       "y": bt["y_train"]})
    for c in exog_cols:
        df[c] = bt["exog_train"][c]
    return df


def origin_nixtla_frames(od: Dict, exog_cols: Optional[List[str]] = None):
    """``(hist_df, futr_df)`` re-anchored to an origin, from an ``origin_eval_data``
    dict.

    ``hist_df`` is the realised conditioning series (the fixed window plus the
    realised gap since the block's refit); ``futr_df`` carries the known day-ahead
    covariates.  Passed to ``NeuralForecast.predict`` so the **frozen** weights and
    the block's conformal widths are reused -- only the conditioning series changes.
    """
    exog_cols = exog_cols or EXOG_COLS
    hist_df = pd.DataFrame({"unique_id": "series_1", "ds": od["ds_history"],
                            "y": od["y_history"]})
    futr_df = pd.DataFrame({"unique_id": "series_1", "ds": od["ds_fut"]})
    for c in exog_cols:
        hist_df[c] = od["exog_history"][c]
        futr_df[c] = od["exog_future"][c]
    return hist_df, futr_df


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
def save_results(df: pd.DataFrame, market: str, model: str,
                 out_dir: str = "./Results") -> str:
    """Persist per-origin metrics for a (market, model) v2 run."""
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{market}_{model}_v2_forecast_eval.csv")
    df.to_csv(fname, index=False)
    return fname
