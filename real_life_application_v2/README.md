# Real-life EPF application — v2 (periodic recalibration)

Companion to `../real_life_application` (v1). Same five markets (NP, PJM, FR, BE,
DE), same seven models (BBEATSx, BART-on-lags, N-BEATSx, N-HiTS, TSMixerX,
Chronos-2, TimesFM), same metric battery — but a **different evaluation protocol**.

## v1 vs v2

| | v1 (`../real_life_application`) | v2 (here) |
|---|---|---|
| Recalibration | refit **every** day-ahead origin | refit **every ~6 months** (`RECAL_BLOCK_DAYS=182`) |
| Fits per (model, market) | ~728 | **4** |
| Training window | rolling | **fixed rolling window** (constant length, oldest days dropped) |
| Within-block forecasts | — | **frozen parameters, re-anchored** to each origin |
| Test origins evaluated | demo: first 28 | **full ~728** |

The fixed rolling window defaults to each market's **original initial training
span** (data start → test start), held constant and rolled forward at each recal
(`utils_epf_v2.default_train_window`).

## Re-anchoring (the core idea)

Within a block a model is **not** refit. Each day-ahead origin is forecast with the
parameters frozen at the block's training, but conditioned on the **realised
series observed up to that origin** (the training window + the realised gap since
the refit). Per model:

- **BBEATSx** — `BBEATSx.forecast_from_origin(y_history, H, exog_future, exog_history)`:
  trend/seasonal/generic evaluated at the origin's absolute time indices with the
  frozen draws; the SV log-volatility is **filtered forward** through the realised
  gap so intervals reflect the volatility at the origin (validated in
  `tests/test_forecast_from_origin.py`).
- **BART-on-lags** — frozen trees; lag features rebuilt from `y_history`; conformal
  `Q` computed once per block.
- **N-BEATSx / N-HiTS / TSMixerX** — `neuralforecast` fit once per block; per-origin
  `predict` on the rolling context window (frozen weights); conformal intervals
  from the block fit.
- **Chronos-2 / TimesFM** — zero-shot; predict per origin on the realised context.

## Parallelism

The 4 blocks are independent (fit once, then re-anchor). Run one block per Colab
runtime via `BLOCK_ID = 0..3`. Every origin is checkpointed and runs resume on
restart. Per-block CSVs concatenate into the full evaluation.

## Files

`{MARKET}/{MARKET}_{model}.ipynb` — 35 notebooks (5 markets × 7 models), all with
the same layout (title · Colab setup · imports · model-family helpers · protocol
parameters + block-recalibration driver). Results are written to
`{MARKET}_{model}_v2_{all|blockN}_forecast_eval.csv`.

All 35 are emitted by **`_build_v2_notebooks.py`** so the study stays uniform — edit
the per-family template there and re-run `python _build_v2_notebooks.py` to
regenerate. The shared slicing / frame / metric helpers live in `utils_epf_v2.py`
(which reuses the v1 loaders and metric battery from `../real_life_application`).
