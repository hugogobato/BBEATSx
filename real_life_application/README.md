# BBEATSx — Real-Life Application: Day-Ahead Electricity Price Forecasting

This folder applies **BBEATSx** and six benchmarks to the five day-ahead
electricity markets used in the original **NBEATSx** paper (Olivares et al.,
*IJF* 2023), loaded straight from the open
[EPFtoolbox database](https://zenodo.org/records/4624805): **NP** (Nord Pool),
**PJM**, **EPEX-BE**, **EPEX-FR**, **EPEX-DE**. Each market has 6 years of hourly
`Price` + two day-ahead exogenous forecasts (`Exogenous 1`, `Exogenous 2`).

## Layout — per-market, per-model notebooks

```
real_life_application/
  utils_epf.py            # data loader + rolling-origin protocol + metric battery
  <MARKET>/
    <MARKET>_bbeatsx.ipynb       # our model (TPU) (explore TPU use)
    <MARKET>_bart_on_lags.ipynb  # benchmark (CPU)
    <MARKET>_nbeatsx.ipynb       # benchmark (GPU)
    <MARKET>_nhits.ipynb         # benchmark (GPU)
    <MARKET>_tsmixerx.ipynb      # benchmark (GPU)
    <MARKET>_chronos2.ipynb      # benchmark (GPU, zero-shot)
    <MARKET>_timesfm.ipynb       # benchmark (GPU, zero-shot)
```

`<MARKET>` ∈ {`NP`, `PJM`, `BE`, `FR`, `DE`} → **35 notebooks**. Each is
self-contained: open it in Colab and run top-to-bottom. The first cell clones the
repo, installs the model's dependencies, and `cd`s here so `utils_epf` is
importable. Results are written to `Results/<MARKET>_origins<a>-<b>_<model>_forecast_eval.csv`.

## Forecasting protocol

Rolling-origin **day-ahead (24h)** forecasting. For each evaluated test day the
model is **refit on a trailing 1-year window** and forecasts the next 24 hours;
the two exogenous covariates are day-ahead forecasts, hence supplied at forecast
time. Every model uses the **same window, origins, covariates and 95% level** so
the comparison is controlled. Top-of-notebook knobs:

| Knob | Default | Meaning |
|---|---|---|
| `TRAIN_WINDOW_DAYS` | `364` | trailing window (set `10000` ⇒ expanding window) |
| `ORIGIN_START` | `0` | first 0-based test day this notebook evaluates |
| `N_ORIGINS` | `28` | number of day-ahead origins (`728` = full 2-yr test set) |
| `RECAL_EVERY` | `1` | refit every K test days |

**Parallelise across Colab runtimes** by giving each copy a disjoint origin chunk
(e.g. `ORIGIN_START=0,N_ORIGINS=30` in one, `ORIGIN_START=30,N_ORIGINS=30` in the
next, …) — exactly the "part" pattern used for the simulation NN benchmarks. The
CSV filename encodes the chunk, so they never collide; concatenate the parts at
the end.

> **Important — why we re-run the benchmarks instead of citing the paper.**
> The NBEATSx paper trains on **~4 years** (expanding, daily recalibration). The
> demo default here is a **1-year** window for BBEATSx tractability. Because the
> window differs, the paper's published numbers are **not** a controlled
> comparison — so every benchmark is re-run here under the *same* protocol. (To
> reproduce the paper's setting exactly, set `TRAIN_WINDOW_DAYS=10000` and
> `N_ORIGINS=728` for **all** models.)

## Hyperparameters

**BBEATSx** uses the **same hyperparameters as the simulation study** — only the
data-structural settings change for hourly EPF data:

- spline trend, **stochastic-volatility (`sv`) errors** (price heteroscedasticity
  / spikes), asymmetric component prior, `num_gfr=50 / num_burnin=200 /
  num_mcmc=1000`, 95% native Bayesian intervals (never conformalised);
- seasonal periods `[(24, 3), (168, 2)]` (daily + weekly Fourier), AR lags
  `(1, 24, 168)`, exogenous `["Exogenous 1", "Exogenous 2"]`;
- `num_threads = os.cpu_count()` — see hardware note below.

Benchmarks mirror their `simulations/04_*` configurations (point NNs + BART use
split-conformal 95% intervals; foundation models are zero-shot). **Caveats:**
Chronos-2 uses the covariates; **TimesFM 2.5 is univariate** (price history only).

## Hardware & threading

BBEATSx and BART-on-lags are **CPU/MCMC** (stochtree) — a **GPU does not
accelerate them.** They scale with vCPU count via `num_threads`:

- free T4 runtime ≈ **2 vCPUs**;
- a **TPU runtime is effectively a many-vCPU host** (`os.cpu_count()` ≈ 24) — use
  it for the two CPU models to parallelise the forest sampler (stochtree ignores
  the TPU itself, but happily uses all the host cores). Speedup is **sublinear**.

Use **GPU (T4)** runtimes for the five neural/foundation models.

## Approximate runtime (1-yr window, 28 origins/market; ±~2×)

| Model | Per origin | Per market | Hardware |
|---|---|---|---|
| BBEATSx (sv) | ~12 min (2 thr) | ~5.5 h | CPU / many-CPU |
| BART-on-lags | ~13 min | ~6 h | CPU / many-CPU |
| NBEATSx | ~2 min | ~1 h | T4 |
| NHITS | ~1.5 min | ~45 min | T4 |
| TSMixerx | ~2 min | ~1 h | T4 |
| Chronos-2 | ~2 s | ~5 min | T4 |
| TimesFM 2.5 | ~2 s | ~5 min | T4 |

On a 24-vCPU host the two CPU models run several-fold faster than the 2-vCPU
figures above.

## Metrics

Per origin: `rmse`, `smape`, `mase`, `crps`, `pinball`, `coverage` (95%),
`interval_score` — identical to the simulation study (`simulations/utils.py`).
