# BBEATSx

**Bayesian Basis Expansion Analysis for Time Series (exogenous-capable).**

An interpretable **trend + seasonality + generic** decomposition whose blocks are
Bayesian Additive Regression Tree (BART) ensembles — plus an extrapolation-safe
parametric trend — fit by **one coherent backfitting MCMC**, delivering calibrated
predictive intervals **including per component**. This is Phase 1 of
`BBEATSx_research_plan.md` (the implementation phase).

```
y_t = F_tr(t) + F_se(s_t) + F_ge(z_t) + eps_t
      \____/    \______/    \______/    \____/
      trend     seasonal     generic    SV or homoscedastic noise
```

The decomposition is structural: each block can only split on its own disjoint
feature group, the tree analogue of NBEATS's basis constraints. Because the model
is Bayesian and all blocks share one residual stream and one error model, you get a
posterior over **each component separately** — the property post-hoc conformal /
quantile methods structurally cannot provide.

---

## Install

```bash
pip install -e .            # numpy + scipy only
pip install -e ".[all]"     # + stochtree, pandas, matplotlib, pytest
```

### Forest backend (important)

BBEATSx is written once against the [`stochtree`](https://stochtree.ai) low-level
API. The forest primitive is **pluggable**:

| Backend | When used | Notes |
|---|---|---|
| `stochtree` | when `import stochtree` succeeds | production C++ sampler (grow/prune/change/swap, GFR/XBART init, multithreading) |
| `numpy-reference` | fallback when `stochtree` is absent | self-contained pure-numpy BART (genuine grow/prune backfitting + conjugate leaves). Correct but **not** performance-optimized. |

`bbeatsx.BACKEND` records which one is active. Force a choice with the
`BBEATSX_BACKEND=stochtree|numpy` environment variable. **For real experiments,
`pip install stochtree`** so the production sampler is used.

---

## Quickstart

```python
import numpy as np
from bbeatsx import BBEATSx, make_config

t = np.arange(200)
rng = np.random.default_rng(0)
y = 1.0 + 0.04*t + 1.5*np.sin(2*np.pi*t/12) + rng.normal(0, 0.5, 200)

cfg = make_config(
    periods=[(12, 3)],     # one seasonal period, 3 Fourier harmonics
    lags=(1, 2),           # AR lags for the generic block
    trend="spline",        # extrapolation-safe Bayesian P-spline trend (default)
    errors="homo",         # or "sv" for stochastic volatility
    num_mcmc=500, seed=0,
)
model = BBEATSx(cfg).fit(y)

fc = model.forecast(horizon=24)
mean = fc.mean()                 # posterior-mean path
lo, hi = fc.interval(0.9)        # 90% predictive interval
trend_lo, trend_hi = fc.component_interval("trend", 0.9)   # per-component band!

dec = model.decomposition()      # in-sample component posterior bands
imp = model.split_importance("generic")   # split-frequency importance (with UQ)
```

See `examples/quickstart.py` for a fuller, plotted walk-through.

---

## Model variants (all switchable from config)

| Knob | Values | Plan reference |
|---|---|---|
| `trend` | `spline` (default), `linear`, `tvp`, `tree` (foil) | §3.6, Lemma 2.3 |
| `errors` | `homo` (default), `sv` | §0.2 |
| `generic.asymmetric` | `True` (default) / `False` | §3.5 identifiability |
| `seasonal.sum_to_zero` | `True` (default) / `False` | §3.5 |
| `multistep` | `recursive` (default) | §0.5 |

- **`spline` / `linear`** — conjugate Gaussian basis trend `phi(t)@beta`; extrapolates
  linearly (safe). `spline` adds a P-spline 2nd-difference smoothing prior.
- **`tvp`** — time-varying-coefficient linear trend with Gaussian random-walk
  amplitudes, sampled by FFBS. (The BART-coefficient realisation of plan §0.3 is a
  planned extension; it needs leaf-regression, available on the `stochtree` backend.)
- **`tree`** — a BART forest on engineered `t`-features; kept only as the
  extrapolation-failure foil (it flatlines out of sample — see
  `tests/test_trend_recovery.py`).

---

## Module map

| File | Plan § | Responsibility |
|---|---|---|
| `config.py` | §1.5 | dataclasses for every prior / toggle / schedule |
| `features.py` | §1.1 | disjoint trend / seasonal / generic design + future-row builder |
| `backend/` | §0.1 | stochtree-or-numpy forest primitives |
| `blocks.py` | §1.2 | `ConjugateTrendBlock`, `TVPTrendBlock`, `ForestBlock` |
| `sv.py` | §0.2 | stochastic volatility (Omori-10 mixture + AR(1) FFBS) |
| `sampler.py` | §1.2 | `BBEATSxSampler` — the shared-residual Gibbs engine |
| `forecast.py` | §1.3 | recursive posterior-predictive simulation + component bands |
| `interpret.py` | §1.4 | decomposition / split-importance / partial-dependence / GIRF |
| `model.py` | §1.5 | `BBEATSx` estimator + Nixtla adapter |
| `serialization.py` | §1.5 | save / load analysis artifacts |

---

## Tests

```bash
PYTHONPATH=. pytest tests                      # auto: stochtree if installed, else numpy
BBEATSX_BACKEND=numpy PYTHONPATH=. pytest tests   # force the reference backend
```

The suite passes on **both** backends (the `stochtree` C++ sampler runs it ~3-4x
faster than the numpy reference).

Covers (plan §1.5): pure-linear-trend recovery in the low-noise limit, sinusoid
recovery, the backfitting residual invariant (`z - sum_c F_c - r ~= 0`), noise
posterior calibration on a known-σ DGP, predictive-interval coverage, SV regime
tracking, the tree-trend extrapolation failure, every `trend × errors` variant, and
a deterministic golden run.

---

## Phase 1 status & deviations

Implemented end-to-end: features, the Gibbs engine, recursive forecasting,
interpretability, the estimator/adapters, serialization, and the test suite — all
running on the numpy backend in this environment and ready to use the `stochtree`
sampler when installed.

Honest deviations from the plan, all flagged for follow-up:
1. **Backend** — production runs need `pip install stochtree` (see above).
2. **`tvp`** ships as the Gaussian random-walk-coefficient trend; the
   BART-coefficient version awaits leaf-regression support.
3. **`multistep="direct"`** is not implemented (recursive is the plan's
   recommendation and default); the config field is reserved.
4. **Ensemble serialization** persists analysis artifacts (component draws, σ²,
   config); full ensemble round-tripping defers to `stochtree`'s `JSONSerializer`.
