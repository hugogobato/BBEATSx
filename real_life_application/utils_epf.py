"""Shared helpers for the BBEATSx real-life electricity-price-forecasting study.

This is the EPF counterpart of ``simulations/utils.py``.  It loads the five
day-ahead electricity markets used in the original NBEATSx paper
(Olivares et al., 2023) -- ``NP``, ``PJM``, ``EPEX-BE``, ``EPEX-FR``,
``EPEX-DE`` -- straight from the open EPFtoolbox Zenodo database
(https://zenodo.org/records/4624805), exposes a rolling-origin **day-ahead
(24h)** forecasting protocol with a configurable training window, and reuses the
*exact* metric battery of the simulation study so BBEATSx and every benchmark are
scored on a comparable footing.

The loader reproduces ``epftoolbox.data.read_data`` (same Zenodo source, same
``Price`` / ``Exogenous 1`` / ``Exogenous 2`` column naming), inlined here so the
notebooks need no ``epftoolbox`` install (and avoid its heavy TF/Keras deps).

Notes on the protocol
----------------------
* Day-ahead means **one refit per evaluated origin**: BBEATSx (and the BART and
  neural benchmarks) forecast recursively from the end of their training window,
  so to condition on the actual prices up to each forecast origin we slide the
  window and refit -- there is no "update history without refit" shortcut.
* The two exogenous covariates are *day-ahead forecasts* (load / generation),
  hence known into the future: they are supplied at forecast time exactly like
  ``exog_future`` in the simulation notebooks.
* The paper trains on ~4 years (expanding, daily recalibration).  The demo
  default here is a **1-year rolling window** for tractability; set a very large
  ``TRAIN_WINDOW_DAYS`` (e.g. 10000) to recover the expanding-window protocol.
  Whatever window you pick, use the *same* one for every model so the comparison
  stays controlled (see README.md).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Datasets (test-period start dates from the NBEATSx paper, section 4.0).      #
# --------------------------------------------------------------------------- #
# market -> (begin_test_date, end_test_date), day-first strings.
MARKET_TEST_DATES: Dict[str, tuple] = {
    "NP":  ("27-12-2016", "24-12-2018"),
    "PJM": ("27-12-2016", "24-12-2018"),
    "FR":  ("04-01-2015", "31-12-2016"),
    "BE":  ("04-01-2015", "31-12-2016"),
    "DE":  ("04-01-2016", "31-12-2017"),
}

EXOG_COLS: List[str] = ["Exogenous 1", "Exogenous 2"]
HORIZON: int = 24           # day-ahead, hourly data
FREQ: str = "h"             # neuralforecast / pandas hourly frequency alias
ZENODO_URL = "https://zenodo.org/records/4624805/files/"

# Quantile levels used to reconstruct a predictive sample from the conformal /
# native interval columns of the neural and foundation benchmarks (matches the
# simulation benchmarks exactly).
NF_LEVELS: List[int] = [50, 60, 70, 80, 90, 95, 98]


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def load_market(market: str, cache_dir: str = "./epf_data") -> pd.DataFrame:
    """Load one market's full hourly series (Price + 2 exogenous covariates).

    Mirrors ``epftoolbox.data.read_data``: reads a local cache if present, else
    downloads ``<market>.csv`` from the EPFtoolbox Zenodo database and caches it.
    """
    if market not in MARKET_TEST_DATES:
        raise ValueError(f"unknown market {market!r}; choose from "
                         f"{list(MARKET_TEST_DATES)}")
    os.makedirs(cache_dir, exist_ok=True)
    fp = os.path.join(cache_dir, market + ".csv")
    if os.path.exists(fp):
        data = pd.read_csv(fp, index_col=0)
    else:
        data = pd.read_csv(ZENODO_URL + market + ".csv", index_col=0)
        data.to_csv(fp)
    data.index = pd.to_datetime(data.index)
    n_exog = len(data.columns) - 1
    data.columns = ["Price"] + [f"Exogenous {i}" for i in range(1, n_exog + 1)]
    return data


def load_split(market: str, cache_dir: str = "./epf_data"):
    """Return ``(data, test_start_pos, n_test_days)`` for a market.

    ``test_start_pos`` is the integer row index of the first test timestamp
    (the paper's official held-out period start); ``n_test_days`` is the number
    of full day-ahead origins available from there to the end of the series.
    """
    data = load_market(market, cache_dir)
    begin_ts = pd.to_datetime(MARKET_TEST_DATES[market][0], dayfirst=True)
    pos = int(data.index.get_indexer([begin_ts])[0])
    if pos < 0:
        raise ValueError(f"test start {begin_ts} not found in {market} index")
    n_test_days = (len(data) - pos) // HORIZON
    return data, pos, n_test_days


def origin_data(data: pd.DataFrame, test_start_pos: int, day: int,
                train_window: int, horizon: int = HORIZON,
                exog_cols: Optional[List[str]] = None) -> Dict:
    """Slice the rolling training window and the 24h day-ahead block for ``day``.

    Parameters
    ----------
    day : int
        0-based test day; the forecast origin is the start of that test day.
    train_window : int
        Number of trailing hourly observations used for training (a fixed
        rolling window).  A value larger than the available history collapses to
        an expanding window.
    """
    exog_cols = exog_cols or EXOG_COLS
    origin = test_start_pos + day * horizon
    start = max(0, origin - train_window)
    train = data.iloc[start:origin]
    fut = data.iloc[origin:origin + horizon]
    return dict(
        origin=origin,
        y_train=train["Price"].to_numpy(float),
        y_test=fut["Price"].to_numpy(float),
        exog_train={c: train[c].to_numpy(float) for c in exog_cols},
        exog_test={c: fut[c].to_numpy(float) for c in exog_cols},
        ds_train=train.index,
        ds_fut=fut.index,
    )


def make_nixtla_frames(d: Dict, exog_cols: Optional[List[str]] = None):
    """Build (train_df, futr_df) long-format frames for neuralforecast models."""
    exog_cols = exog_cols or EXOG_COLS
    df_train = pd.DataFrame(
        {"unique_id": "series_1", "ds": d["ds_train"], "y": d["y_train"]})
    futr_df = pd.DataFrame({"unique_id": "series_1", "ds": d["ds_fut"]})
    for c in exog_cols:
        df_train[c] = d["exog_train"][c]
        futr_df[c] = d["exog_test"][c]
    return df_train, futr_df


def save_results(df: pd.DataFrame, market: str, model: str,
                 origin_start: int, origin_end: int,
                 out_dir: str = "./Results") -> str:
    """Persist per-origin metrics; filename encodes market + origin chunk."""
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f"{market}_origins{origin_start}-{origin_end}_{model}_forecast_eval.csv")
    df.to_csv(fname, index=False)
    return fname


# --------------------------------------------------------------------------- #
# Predictive-sample reconstruction (shared by NN + foundation benchmarks)     #
# --------------------------------------------------------------------------- #
def samples_from_quantiles(Q: np.ndarray, q_probs: np.ndarray,
                           n_samples: int = 200) -> np.ndarray:
    """Inverse-CDF reconstruction of a predictive sample ``(H, n_samples)`` from
    a quantile-value matrix ``Q (H, n_q)`` at probabilities ``q_probs``."""
    order = np.argsort(q_probs)
    q_probs = np.asarray(q_probs)[order]
    Q = np.asarray(Q)[:, order]
    Q = np.maximum.accumulate(Q, axis=1)                  # monotone quantiles
    probs = (np.arange(n_samples) + 0.5) / n_samples      # stratified, deterministic
    return np.vstack([np.interp(probs, q_probs, Q[h]) for h in range(Q.shape[0])])


def samples_from_nf(res: pd.DataFrame, model_name: str, horizon: int,
                    levels: Optional[List[int]] = None,
                    n_samples: int = 200) -> np.ndarray:
    """Reconstruct a predictive sample from a neuralforecast prediction frame
    carrying ``-lo-L`` / ``-hi-L`` interval columns at the given ``levels``."""
    levels = levels or NF_LEVELS
    q_cols = {0.5: model_name}
    for L in levels:
        a = (1.0 - L / 100.0) / 2.0
        q_cols[a] = f"{model_name}-lo-{L}"
        q_cols[1.0 - a] = f"{model_name}-hi-{L}"
    q_probs = np.array(sorted(q_cols))
    Q = np.column_stack([res[q_cols[p]].values for p in q_probs])  # (H, n_q)
    return samples_from_quantiles(Q, q_probs, n_samples)


# --------------------------------------------------------------------------- #
# Metric battery (verbatim from simulations/utils.py for cross-study parity)   #
# --------------------------------------------------------------------------- #
def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(predicted)
    return float(np.mean(200.0 * np.abs(actual - predicted) / np.maximum(denom, 1e-8)))


def compute_mase(actual: np.ndarray, predicted: np.ndarray, y_train: np.ndarray) -> float:
    mae = np.mean(np.abs(actual - predicted))
    scale = np.mean(np.abs(np.diff(y_train)))
    return float(mae / np.maximum(scale, 1e-8))


def compute_crps(actual: np.ndarray, samples: np.ndarray) -> float:
    """CRPS for a sample-based predictive distribution. actual (H,), samples (H,S)."""
    H, S = samples.shape
    term1 = np.mean(np.abs(samples - actual[:, None]), axis=1)
    samples_sorted = np.sort(samples, axis=1)
    coef = 2.0 * np.arange(1, S + 1) - S - 1
    term2 = np.sum(coef[None, :] * samples_sorted, axis=1) / (S ** 2)
    return float(np.mean(term1 - term2))


def compute_pinball_loss(actual: np.ndarray, samples: np.ndarray,
                         quantiles: np.ndarray = None) -> float:
    if quantiles is None:
        quantiles = np.linspace(0.05, 0.95, 19)
    q_vals = np.quantile(samples, quantiles, axis=1)  # (n_q, H)
    loss = 0.0
    for i, q in enumerate(quantiles):
        diff = actual - q_vals[i]
        loss += np.mean(np.maximum(q * diff, (q - 1) * diff))
    return float(loss / len(quantiles))


def compute_coverage(actual: np.ndarray, samples: np.ndarray, level: float = 0.9) -> float:
    a = (1.0 - level) / 2.0
    lo = np.quantile(samples, a, axis=1)
    hi = np.quantile(samples, 1.0 - a, axis=1)
    return float(np.mean((actual >= lo) & (actual <= hi)))


def compute_interval_score(actual: np.ndarray, samples: np.ndarray, level: float = 0.9) -> float:
    alpha = 1.0 - level
    a = alpha / 2.0
    lo = np.quantile(samples, a, axis=1)
    hi = np.quantile(samples, 1.0 - a, axis=1)
    score = (hi - lo) + (2.0 / alpha) * (lo - actual) * (actual < lo) \
        + (2.0 / alpha) * (actual - hi) * (actual > hi)
    return float(np.mean(score))


def evaluate_samples(actual: np.ndarray, samples: np.ndarray, y_train: np.ndarray,
                     level: float = 0.95, point: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Full point + probabilistic metric battery for one 24h day-ahead forecast.

    ``samples`` is ``(H, S)``.  ``point`` overrides the point forecast used for
    RMSE/sMAPE/MASE (e.g. a model's native mean); defaults to the sample mean.
    """
    mean_pred = np.asarray(point) if point is not None else samples.mean(axis=1)
    return {
        "rmse": compute_rmse(actual, mean_pred),
        "smape": compute_smape(actual, mean_pred),
        "mase": compute_mase(actual, mean_pred, y_train),
        "crps": compute_crps(actual, samples),
        "pinball": compute_pinball_loss(actual, samples),
        "coverage": compute_coverage(actual, samples, level=level),
        "interval_score": compute_interval_score(actual, samples, level=level),
    }
