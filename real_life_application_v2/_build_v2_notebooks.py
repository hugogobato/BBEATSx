"""Generate the v2 (periodic-recalibration) notebooks for every (market, model).

Five markets x seven models = 35 notebooks, all emitted with an identical layout
so the v2 study stays uniform:

    cell 0  markdown title / protocol note
    cell 1  Colab setup (clone + per-model pip install, cd, sys.path)
    cell 2  imports (+ any model/pipeline load)
    cell 3  model-family helpers (BART trees, neural fit/predict, foundation run)
    cell 4  v2 protocol parameters + the block-recalibration driver

The driver differs only by model family:
  * BBEATSx     -- fit once per block, ``forecast_from_origin`` per origin.
  * BART-on-lags-- fit trees + conformal once per block, roll frozen trees + reuse
                   the block conformal widths per re-anchored origin.
  * neural      -- ``neuralforecast`` fit once per block, ``predict`` on the rolling
                   context per origin (frozen weights + block conformal widths).
  * foundation  -- zero-shot, predict per origin on the realised context (no refit).

Re-run with ``python _build_v2_notebooks.py`` to regenerate after editing a template.
"""
from __future__ import annotations

import json
import os

MARKETS = ["PJM", "BE", "DE", "FR", "NP"]
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# notebook plumbing                                                           #
# --------------------------------------------------------------------------- #
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n") + "\n"}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n") + "\n"}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def sub(text, **kw):
    for k, v in kw.items():
        text = text.replace("__%s__" % k, v)
    return text


# --------------------------------------------------------------------------- #
# shared cell fragments                                                       #
# --------------------------------------------------------------------------- #
SETUP = """
# Google Colab setup ------------------------------------------------------------
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import os, sys
if IN_COLAB:
    print("Running in Google Colab. Setting up workspace...")
    if not os.path.abspath(".").endswith("real_life_application_v2"):
        if not os.path.exists("BBEATSx"):
            !git clone https://github.com/hugogobato/BBEATSx.git
        !pip install -q ./BBEATSx
__PIP__
        %cd BBEATSx/real_life_application_v2
    else:
        print("Already in real_life_application_v2 directory.")
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../real_life_application"))   # shared v1 loaders/metrics
"""

# Common header: protocol params + output/resume scaffolding. __THREADS__ and the
# block loop body (__LOOP__) are filled per family.
DRIVER = """
# ---- v2 protocol parameters --------------------------------------------------
MARKET           = "__MARKET__"
MODEL_TAG        = "__MODEL_TAG__"
LEVEL            = 0.95
HORIZON          = U2.HORIZON
RECAL_BLOCK_DAYS = U2.RECAL_BLOCK_DAYS     # ~6 months -> ~4 refits over the test set
__THREADS__
data, test_start_pos, n_test_days = U2.load_split(MARKET)
# Fixed rolling window = the market's original initial training span (data start
# -> test start), held constant and rolled forward at each recalibration.
TRAIN_WINDOW = U2.default_train_window(test_start_pos)
blocks = U2.recal_blocks(n_test_days, RECAL_BLOCK_DAYS)
print(f"{MARKET}: {n_test_days} test days | {len(blocks)} recalibration blocks "
      f"(every {RECAL_BLOCK_DAYS} days) | rolling window = {TRAIN_WINDOW // 24} days")

# ---- optional block-level parallelism across Colab runtimes ------------------
# Each block is independent (fit once, then re-anchor its origins): run one block
# per runtime by setting BLOCK_ID; leave None to run every block in this notebook.
BLOCK_ID = None            # or 0, 1, 2, 3 to run a single block on this runtime
run_blocks = blocks if BLOCK_ID is None else [blocks[BLOCK_ID]]

# ---- output + crash-safe resume ---------------------------------------------
USE_DRIVE = True
if IN_COLAB and USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped:", e)
OUT_DIR = ("/content/drive/MyDrive/BBEATSx_Results_v2"
           if os.path.isdir("/content/drive/MyDrive") else "./Results")
os.makedirs(OUT_DIR, exist_ok=True)
tag = "all" if BLOCK_ID is None else f"block{BLOCK_ID}"
fname = os.path.join(OUT_DIR, f"{MARKET}_{MODEL_TAG}_v2_{tag}_forecast_eval.csv")

rows, done = [], set()
if os.path.exists(fname):                       # resume an interrupted run
    prev = pd.read_csv(fname); rows = prev.to_dict("records")
    done = set(prev["test_day"].astype(int))
    print(f"Resuming: {len(done)} origins already saved.")

__LOOP__

df = pd.DataFrame(rows)
print("\\nSaved:", fname)
print(df.drop(columns=["origin"]).mean(numeric_only=True).to_frame("Mean Metric"))
"""

# Per-origin evaluate + atomic-checkpoint tail (shared by every family). Expects
# ``samples`` (H, S) and ``y_hat`` (H,) to be defined for the current origin.
SAVE_TAIL = """            m = U2.evaluate_samples(od["y_test"], samples, bt["y_train"],
                                    level=LEVEL, point=y_hat)
            m["test_day"] = day; m["block"] = bstart
            m["origin"] = str(od["ds_fut"][0])
            rows.append(m)
            tmp = fname + ".tmp"; pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, fname)
            print(f"  day {day:4d} ({od['ds_fut'][0].date()}): "
                  f"rmse={m['rmse']:.3f}  cov95={m['coverage']:.2f}")
        except Exception as e:
            print(f"  day {day} failed: {e}")"""


# --------------------------------------------------------------------------- #
# per-family block loops                                                      #
# --------------------------------------------------------------------------- #
LOOP_BBEATSX = """for (bstart, bend) in run_blocks:
    bt = U2.block_train_data(data, test_start_pos, bstart, TRAIN_WINDOW)
    if not all(d in done for d in range(bstart, bend)):     # skip refit if block done
        cfg = make_config(
            periods=[(24, 3), (168, 2)], lags=(1, 24, 168), exog=U2.EXOG_COLS,
            trend="spline", errors="sv", asymmetric=True,
            num_gfr=50, num_burnin=200, num_mcmc=1000,
            num_threads=NUM_THREADS, seed=bstart)
        print(f"\\nblock {bstart}..{bend}: fitting on {len(bt['y_train']) // 24} days "
              f"ending {bt['ds_train'][-1].date()} ...")
        model = BBEATSx(cfg).fit(bt["y_train"], exog=bt["exog_train"])
    for day in range(bstart, bend):
        if day in done:
            continue
        od = U2.origin_eval_data(data, test_start_pos, bt["window_start_abs"], day, HORIZON)
        try:
            fc = model.forecast_from_origin(
                od["y_history"], HORIZON,
                exog_future=od["exog_future"], exog_history=od["exog_history"])
            samples, y_hat = fc.samples, fc.mean()
__SAVE__"""

LOOP_BART = """for (bstart, bend) in run_blocks:
    bt = U2.block_train_data(data, test_start_pos, bstart, TRAIN_WINDOW)
    if not all(d in done for d in range(bstart, bend)):     # fit trees + conformal once
        print(f"\\nblock {bstart}..{bend}: fitting BART + conformal on "
              f"{len(bt['y_train']) // 24} days ending {bt['ds_train'][-1].date()} ...")
        Q = conformal_thresholds(bt["y_train"], bt["exog_train"], HORIZON, n_windows=5)
        bart = fit_bart(bt["y_train"], bt["exog_train"])
    for day in range(bstart, bend):
        if day in done:
            continue
        od = U2.origin_eval_data(data, test_start_pos, bt["window_start_abs"], day, HORIZON)
        try:
            y_hat = roll_bart(bart, od["y_history"], od["exog_future"], HORIZON)
            sigma = Q / 1.96
            samples = np.random.normal(y_hat[:, None], sigma[:, None], size=(HORIZON, 200))
__SAVE__"""

LOOP_NEURAL = """for (bstart, bend) in run_blocks:
    bt = U2.block_train_data(data, test_start_pos, bstart, TRAIN_WINDOW)
    if not all(d in done for d in range(bstart, bend)):     # fit once per block
        print(f"\\nblock {bstart}..{bend}: fitting {MODEL_NAME} on "
              f"{len(bt['y_train']) // 24} days ending {bt['ds_train'][-1].date()} ...")
        nf = fit_block(bt, HORIZON)
    for day in range(bstart, bend):
        if day in done:
            continue
        od = U2.origin_eval_data(data, test_start_pos, bt["window_start_abs"], day, HORIZON)
        try:
            samples, y_hat = predict_origin(nf, od, HORIZON)
__SAVE__"""

LOOP_FOUNDATION = """for (bstart, bend) in run_blocks:
    bt = U2.block_train_data(data, test_start_pos, bstart, TRAIN_WINDOW)   # window only
    print(f"\\nblock {bstart}..{bend}: zero-shot over {bend - bstart} origins "
          f"(context window = {TRAIN_WINDOW // 24} days, no refit) ...")
    for day in range(bstart, bend):
        if day in done:
            continue
        od = U2.origin_eval_data(data, test_start_pos, bt["window_start_abs"], day, HORIZON)
        try:
            samples, y_hat = run_model(od, HORIZON)
__SAVE__"""


# --------------------------------------------------------------------------- #
# per-family helper cells                                                     #
# --------------------------------------------------------------------------- #
HELP_BART = """
# ---- BART-on-lags: frozen trees re-anchored per origin, block conformal ------
BART_LAGS = 168   # week of hourly lags (covers BBEATSx's 1/24/168 reach)


def _bart_design(y, exog, lags):
    X = np.column_stack([np.roll(y, lag) for lag in range(1, lags + 1)])[lags:]
    if exog is not None:
        X = np.column_stack([X] + [v[lags:] for v in exog.values()])
    return X, y[lags:]


def fit_bart(y_train, exog_train, lags=BART_LAGS):
    \"\"\"Fit BART once on the block's training window (frozen for the whole block).\"\"\"
    X_train, y_target = _bart_design(y_train, exog_train, lags)
    model = BARTModel()
    model.sample(
        X_train=X_train, y_train=y_target,
        num_gfr=50, num_burnin=200, num_mcmc=1000,
        general_params={"sample_sigma2_global": False},
        mean_forest_params={"num_trees": 50, "sample_sigma2_leaf": False},
        variance_forest_params={"num_trees": 10},
    )
    return model


def roll_bart(model, y_history, exog_future, horizon, lags=BART_LAGS):
    \"\"\"Recursively roll the frozen trees H steps from the realised history.\"\"\"
    history = list(y_history[-lags:])
    y_hat = []
    for step in range(horizon):
        current_lags = np.array(history[-lags:][::-1])
        if exog_future is not None:
            exog_step = [v[step] for v in exog_future.values()]
            X_step = np.concatenate([current_lags, exog_step]).reshape(1, -1)
        else:
            X_step = current_lags.reshape(1, -1)
        preds = model.predict(X_step)
        yh = float(np.mean(preds["mean_forest_predictions"].flatten()))
        y_hat.append(yh); history.append(yh)
    return np.array(y_hat)


def fit_predict_bart(y_train, exog_train, exog_test, horizon, lags=BART_LAGS):
    \"\"\"Fit + roll in one call (used only inside the per-block conformal calibration).\"\"\"
    return roll_bart(fit_bart(y_train, exog_train, lags), y_train, exog_test, horizon, lags)


def conformal_thresholds(y_train, exog_train, horizon, n_windows=5):
    \"\"\"Split-conformal H-step thresholds, calibrated once per block.\"\"\"
    n_train = len(y_train)
    errors = [[] for _ in range(horizon)]
    for k in range(n_windows):
        prop = n_train - horizon - n_windows + 1 + k
        y_prop, y_cal = y_train[:prop], y_train[prop:prop + horizon]
        if exog_train is not None:
            ex_prop = {key: val[:prop] for key, val in exog_train.items()}
            ex_cal = {key: val[prop:prop + horizon] for key, val in exog_train.items()}
        else:
            ex_prop = ex_cal = None
        try:
            y_cal_hat = fit_predict_bart(y_prop, ex_prop, ex_cal, horizon)
            for h in range(horizon):
                errors[h].append(np.abs(y_cal[h] - y_cal_hat[h]))
        except Exception as e:
            print(f"    calibration window {k} failed: {e}")
    Q = np.array([np.percentile(errors[h], 95) if errors[h] else 1.0
                  for h in range(horizon)])
    return np.maximum(Q, 1e-3)
"""

HELP_NEURAL = """
# ---- __MODEL_NAME__: fit once per block, predict per re-anchored origin -------
def fit_block(bt, horizon):
    \"\"\"Fit __MODEL_NAME__ once on the block training window, with split-conformal
    prediction intervals calibrated on that window.\"\"\"
    df_train = U2.block_nixtla_frame(bt)
    input_size = max(2 * horizon, 2 * 168)   # 2-week lookback (covers weekly cycle)
    common = dict(
        h=horizon,
        input_size=input_size,
__MODEL_KWARGS__
        futr_exog_list=U2.EXOG_COLS,
        loss=MAE(),
        valid_loss=RMSE(),
        val_monitor="train_loss",
    )
    model = MODEL_CLASS(**common)
    prediction_intervals = PredictionIntervals(n_windows=5, method="conformal_distribution")
    nf = NeuralForecast(models=[model], freq=U2.FREQ)
    nf.fit(df=df_train, val_size=horizon, prediction_intervals=prediction_intervals)
    return nf


def predict_origin(nf, od, horizon):
    \"\"\"Re-anchored forecast: frozen weights + block conformal widths, new context.
    ``predict(df=...)`` swaps in the realised conditioning series without refitting.\"\"\"
    hist_df, futr_df = U2.origin_nixtla_frames(od)
    res = nf.predict(df=hist_df, futr_df=futr_df, level=U2.NF_LEVELS)
    y_hat = res[MODEL_NAME].values
    samples = U2.samples_from_nf(res, MODEL_NAME, horizon)
    return samples, y_hat
"""

HELP_CHRONOS = """
# ---- Chronos-2: zero-shot, re-anchored to the realised context per origin -----
def _match_q_column(columns, q):
    cols = set(map(str, columns))
    for cand in (str(q), f"{q:g}", f"{q:.3f}", f"{q:.2f}", str(float(q))):
        if cand in cols:
            return cand
    return None


def run_model(od, horizon):
    \"\"\"Zero-shot Chronos-2 day-ahead forecast conditioned on the realised window.\"\"\"
    context_df = pd.DataFrame({"id": "series_1", "timestamp": od["ds_history"],
                               "target": od["y_history"]})
    future_df = pd.DataFrame({"id": "series_1", "timestamp": od["ds_fut"]})
    for c in U2.EXOG_COLS:
        context_df[c] = od["exog_history"][c]
        future_df[c] = od["exog_future"][c]

    pred_df = pipeline.predict_df(
        context_df, future_df=future_df, prediction_length=horizon,
        quantile_levels=QUANTILE_LEVELS,
        id_column="id", timestamp_column="timestamp", target="target",
    ).reset_index(drop=True)
    if "timestamp" in pred_df.columns:
        pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)
    pred_df = pred_df.iloc[:horizon]

    q_probs, q_cols = [], []
    for q in QUANTILE_LEVELS:
        col = _match_q_column(pred_df.columns, q)
        if col is not None:
            q_probs.append(q); q_cols.append(col)
    if not q_cols:
        raise RuntimeError(f"No quantile columns in Chronos output: {list(pred_df.columns)}")

    Q = pred_df[q_cols].to_numpy(dtype=float)
    samples = U2.samples_from_quantiles(Q, np.array(q_probs))
    if "predictions" in pred_df.columns:
        y_hat = pred_df["predictions"].to_numpy(dtype=float)
    else:
        y_hat = pred_df[_match_q_column(pred_df.columns, 0.5)].to_numpy(dtype=float)
    return samples, y_hat
"""

HELP_TIMESFM = """
# ---- TimesFM 2.5: zero-shot univariate, re-anchored to the realised context ---
def run_model(od, horizon):
    \"\"\"Zero-shot TimesFM forecast (univariate; exogenous inputs ignored).\"\"\"
    ctx = torch.tensor(np.asarray(od["y_history"], dtype=np.float32), device=model.device)
    with torch.no_grad():
        outputs = model(past_values=[ctx], return_dict=True)
    mean_pred = np.asarray(outputs.mean_predictions[0].float().cpu().numpy())[:horizon]
    full = np.asarray(outputs.full_predictions[0].float().cpu().numpy())[:horizon]
    nq = len(TFM_QUANTILES)
    Q = full[:, 1:] if full.shape[-1] == nq + 1 else full[:, :nq]
    samples = U2.samples_from_quantiles(Q, TFM_QUANTILES)
    return samples, mean_pred
"""


# --------------------------------------------------------------------------- #
# imports cells                                                               #
# --------------------------------------------------------------------------- #
IMP_PREFIX = """import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
"""
IMP_SUFFIX = """import utils_epf as U
import utils_epf_v2 as U2
print("__MODEL_NAME__ real-life EPF application -- v2 (periodic recalibration)")
"""

IMP_BBEATSX = IMP_PREFIX + "from bbeatsx import BBEATSx, make_config\n" + IMP_SUFFIX
IMP_BART = IMP_PREFIX + "from stochtree import BARTModel\n" + IMP_SUFFIX
IMP_NEURAL = IMP_PREFIX + (
    "from neuralforecast import NeuralForecast\n"
    "from neuralforecast.models import __MODEL_NAME__\n"
    "from neuralforecast.losses.pytorch import MAE, RMSE\n"
    "from neuralforecast.utils import PredictionIntervals\n"
) + IMP_SUFFIX + "\nMODEL_CLASS = __MODEL_NAME__\nMODEL_NAME = \"__MODEL_NAME__\"\n"
IMP_CHRONOS = IMP_PREFIX + (
    "from chronos import Chronos2Pipeline\n"
    "import torch\n"
) + IMP_SUFFIX + (
    "\npipeline = Chronos2Pipeline.from_pretrained(\n"
    "    \"amazon/chronos-2\",\n"
    "    device_map=\"cuda\" if torch.cuda.is_available() else \"cpu\",\n"
    ")\n"
    "QUANTILE_LEVELS = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50,\n"
    "                   0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]\n"
)
IMP_TIMESFM = IMP_PREFIX + (
    "import torch\n"
    "from transformers import TimesFm2_5ModelForPrediction\n"
) + IMP_SUFFIX + (
    "\nmodel = TimesFm2_5ModelForPrediction.from_pretrained(\n"
    "    \"google/timesfm-2.5-200m-transformers\", device_map=\"auto\")\n"
    "model.eval()\n"
    "TFM_QUANTILES = np.array(getattr(model.config, \"quantiles\",\n"
    "                                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))\n"
)


# --------------------------------------------------------------------------- #
# titles                                                                      #
# --------------------------------------------------------------------------- #
V2_NOTE = (
    "**v2 protocol.** Refit every ~6 months (`RECAL_BLOCK_DAYS`) on a **fixed "
    "rolling window**, then forecast every day-ahead origin in the block with the "
    "**frozen** model, re-anchored to the origin by conditioning on the realised "
    "prices observed up to it. Evaluates the full ~728-origin test set at ~4 fits "
    "instead of 728 -- see `../real_life_application` for the v1 daily-refit "
    "protocol. Each block is independent, so you can run one block per Colab "
    "runtime via `BLOCK_ID`; results checkpoint after every origin and resume on "
    "restart."
)

TITLES = {
    "bbeatsx": "# BBEATSx -- Real-Life Application (__MARKET__), v2: periodic recalibration\n\n"
               "The Bayesian BART-block N-BEATSx with spline trend, stochastic-volatility "
               "errors and Fourier seasonality. Re-anchoring uses "
               "`forecast_from_origin`: the frozen posterior draws are evaluated at the "
               "origin's time indices and the SV log-volatility is filtered through the "
               "realised gap so intervals reflect the volatility at the origin.\n\n" + V2_NOTE,
    "bart_on_lags": "# BART-on-lags -- Real-Life Application (__MARKET__), v2: periodic recalibration\n\n"
               "BART regression on a week of hourly price lags + the two day-ahead "
               "covariates, with **split-conformal** 95% intervals (justified for "
               "recursive point predictors -- see `simulations/SIMULATION_ANALYSIS.md` "
               "Section 9). The trees and the conformal thresholds are fit **once per "
               "block**; each origin rolls the frozen trees forward from the realised "
               "history and reuses the block's conformal widths. **CPU model.**\n\n" + V2_NOTE,
    "nbeatsx": "# NBEATSx -- Real-Life Application (__MARKET__), v2: periodic recalibration\n\n"
               "**NBEATSx** (neuralforecast) with **split-conformal** 95% intervals. "
               "Fit **once per block**; each origin re-anchors via `predict` on the "
               "rolling realised context (frozen weights + block conformal widths). "
               "Runs on a **T4 GPU**.\n\n" + V2_NOTE,
    "nhits": "# NHITS -- Real-Life Application (__MARKET__), v2: periodic recalibration\n\n"
               "**NHITS** (neuralforecast) with **split-conformal** 95% intervals. "
               "Fit **once per block**; each origin re-anchors via `predict` on the "
               "rolling realised context (frozen weights + block conformal widths). "
               "Runs on a **T4 GPU**.\n\n" + V2_NOTE,
    "tsmixerx": "# TSMixerx -- Real-Life Application (__MARKET__), v2: periodic recalibration\n\n"
               "**TSMixerx** (neuralforecast) with **split-conformal** 95% intervals. "
               "Fit **once per block**; each origin re-anchors via `predict` on the "
               "rolling realised context (frozen weights + block conformal widths). "
               "Runs on a **T4 GPU**.\n\n" + V2_NOTE,
    "chronos2": "# Chronos-2 -- Real-Life Application (__MARKET__), v2: periodic recalibration\n\n"
               "**Chronos-2** zero-shot foundation model (uses the two day-ahead "
               "covariates via `predict_df`). No training: each origin conditions on the "
               "realised window up to it. Runs on a **T4 GPU**.\n\n" + V2_NOTE,
    "timesfm": "# TimesFM 2.5 -- Real-Life Application (__MARKET__), v2: periodic recalibration\n\n"
               "**TimesFM 2.5** zero-shot foundation model -- **univariate** here "
               "(forecasts price from price history only; exogenous covariates ignored). "
               "Each origin conditions on the realised window up to it. Runs on a "
               "**T4 GPU**.\n\n" + V2_NOTE,
}


# --------------------------------------------------------------------------- #
# per-model assembly spec                                                     #
# --------------------------------------------------------------------------- #
NEURAL_KWARGS = {
    "nbeatsx": "        max_steps=1000,\n        early_stop_patience_steps=-1,   # fixed budget (matches sim setup)\n",
    "nhits": "        max_steps=1000,\n        val_check_steps=10,\n        early_stop_patience_steps=5,\n",
    "tsmixerx": "        max_steps=1000,\n        n_series=1,                          # single price series per market\n        val_check_steps=10,\n        early_stop_patience_steps=5,\n",
}

MODELS = {
    "bbeatsx": dict(model_name="BBEATSx", pip="        !pip install -q stochtree",
                    imports=IMP_BBEATSX, helper=None, loop=LOOP_BBEATSX,
                    threads="NUM_THREADS      = os.cpu_count()          # BBEATSx is CPU/MCMC; the GPU does NOT help\n"),
    "bart_on_lags": dict(model_name="BART-on-lags", pip="        !pip install -q stochtree",
                    imports=IMP_BART, helper=HELP_BART, loop=LOOP_BART, threads=""),
    "nbeatsx": dict(model_name="NBEATSx", pip="        !pip install -q neuralforecast",
                    imports=IMP_NEURAL, helper=HELP_NEURAL, loop=LOOP_NEURAL, threads="",
                    kwargs=NEURAL_KWARGS["nbeatsx"]),
    "nhits": dict(model_name="NHITS", pip="        !pip install -q neuralforecast",
                    imports=IMP_NEURAL, helper=HELP_NEURAL, loop=LOOP_NEURAL, threads="",
                    kwargs=NEURAL_KWARGS["nhits"]),
    "tsmixerx": dict(model_name="TSMixerx", pip="        !pip install -q neuralforecast",
                    imports=IMP_NEURAL, helper=HELP_NEURAL, loop=LOOP_NEURAL, threads="",
                    kwargs=NEURAL_KWARGS["tsmixerx"]),
    "chronos2": dict(model_name="Chronos-2",
                    pip="        !pip install -q \"chronos-forecasting>=2.0\" \"pandas[pyarrow]\"",
                    imports=IMP_CHRONOS, helper=HELP_CHRONOS, loop=LOOP_FOUNDATION, threads=""),
    "timesfm": dict(model_name="TimesFM 2.5", pip="        !pip install -q -U transformers accelerate",
                    imports=IMP_TIMESFM, helper=HELP_TIMESFM, loop=LOOP_FOUNDATION, threads=""),
}


def build(market, model_tag, spec):
    name = spec["model_name"]
    cells = [md(sub(TITLES[model_tag], MARKET=market))]
    cells.append(code(sub(SETUP, PIP=spec["pip"])))
    cells.append(code(sub(spec["imports"], MODEL_NAME=name)))
    if spec["helper"] is not None:
        helper = spec["helper"]
        if "kwargs" in spec:
            helper = sub(helper, MODEL_KWARGS=spec["kwargs"].rstrip("\n"))
        cells.append(code(sub(helper, MODEL_NAME=name)))
    loop = sub(spec["loop"], SAVE=SAVE_TAIL)
    driver = sub(DRIVER, MARKET=market, MODEL_TAG=model_tag,
                 THREADS=spec["threads"].rstrip("\n"), LOOP=loop)
    cells.append(code(driver))
    return notebook(cells)


def main():
    n = 0
    for market in MARKETS:
        out_dir = os.path.join(HERE, market)
        os.makedirs(out_dir, exist_ok=True)
        for model_tag, spec in MODELS.items():
            nb = build(market, model_tag, spec)
            path = os.path.join(out_dir, f"{market}_{model_tag}.ipynb")
            with open(path, "w") as f:
                json.dump(nb, f, indent=1)
            n += 1
    print(f"wrote {n} notebooks across {len(MARKETS)} markets")


if __name__ == "__main__":
    main()
