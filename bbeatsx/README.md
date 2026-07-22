# BBEATSx

**Bayesian Basis Expansion Analysis for Time Series (exogenous-capable).**

An interpretable additive decomposition with an extrapolation-safe parametric
trend and Bayesian Additive Regression Tree (BART) correction blocks, fit by one
shared-residual backfitting sampler. The original layout is **trend + seasonality
+ generic**. An opt-in layout separates the last term into **exogenous +
autoregressive** contributions. Posterior component bands are reported, but their
frequentist calibration is a scientific property to test, not an API guarantee.
This is Phase 1 of
`BBEATSx_research_plan.md` (the implementation phase).

```
y_t = F_tr(t) + F_se(s_t) + F_ge(z_t) + eps_t
      \____/    \______/    \______/    \____/
      trend     seasonal     generic    SV or homoscedastic noise

or, with `generic.component_layout="split"`,

y_t = F_tr(t) + F_se(s_t) + F_x(x_t) + F_ar(y_{t-lags}) + eps_t.
```

The decomposition is feature-routed: each forest can split only on its assigned
columns. This is not the same as statistical identification. In particular, raw
outcome lags contain past trend, seasonality, exogenous signal, and noise, so the
autoregressive block can still compete with the structural blocks.

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
| `stochtree` | when `import stochtree` succeeds | production C++ sampler; recursive GFR warm start followed by grow/prune birth-death MCMC in the validated 0.4.4 release |
| `numpy-reference` | fallback when `stochtree` is absent | experimental pure-NumPy grow/prune kernel with different initialization, split proposals, random-number stream, and optional leaf-variance convention |

The two implementations expose the same adapter API, but they are not known to
target an identical finite-sample transition kernel. `bbeatsx.BACKEND` and
`bbeatsx.BACKEND_VERSION` record which implementation produced a run. Force a
choice with `BBEATSX_BACKEND=stochtree|numpy`. For manuscript experiments use
the pinned `stochtree==0.4.4` backend and set the environment variable explicitly;
use NumPy for tests, debugging, and independent conformance checks.

The version pin matters. BBEATSx uses the low-level variance-weight interface,
whose public documentation and compiled 0.4.4 Gaussian sufficient statistics do
not describe the convention consistently. `tests/test_backend_variance_weights.py`
therefore checks the convention against an analytic one-leaf posterior. Upgrade
`stochtree` only after that contract and the posterior-equivalence tests pass.

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

For separate predictive attribution, pass `component_layout="split"` together
with named exogenous columns. Forecast components will then be `trend`,
`seasonal`, `exogenous`, and `autoregressive`. This static split is opt-in because
it removes exogenous-lag interactions and is not, by itself, an identification or
causal adjustment.

See `examples/quickstart.py` for a fuller, plotted walk-through.

---

## Model variants (all switchable from config)

| Knob | Values | Plan reference |
|---|---|---|
| `trend` | `spline` (default), `linear`, `tvp`, `tree` (foil) | §3.6, Lemma 2.3 |
| `errors` | `homo` (default), `sv` | §0.2 |
| `generic.asymmetric` | `True` (default) / `False` | §3.5 identifiability |
| `generic.component_layout` | `combined` (default), `split` | predictive attribution experiment |
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

The deterministic and analytic contract tests pass on both backends. This does
not establish that their full forest posteriors are equivalent. In an isolated
DGP-1 benchmark at the study schedule, `stochtree` 0.4.4 was about 9.3 times
faster than NumPy; the exact ratio is workload- and machine-dependent. The gain
comes mainly from compiled tree traversal, partition tracking, cached predictions,
and native storage. For small fits, increasing inner `num_threads` did not help,
so parallel simulation workers should normally keep `num_threads=1`.

Covers (plan §1.5): pure-linear-trend recovery in the low-noise limit, sinusoid
recovery, the backfitting residual invariant (`z - sum_c F_c - r ~= 0`), noise
posterior calibration on a known-σ DGP, predictive-interval coverage, SV regime
tracking, the tree-trend extrapolation failure, every `trend × errors` variant, and
a deterministic golden run.

---

## Phase 1 status & deviations

Implemented end-to-end: features, the Gibbs engine, recursive forecasting,
interpretability, the estimator/adapters, serialization, backend provenance, and
the test suite. Both the legacy combined layout and the opt-in static split are
covered by regression tests.

Honest deviations from the plan, all flagged for follow-up:
1. **Backend**: production runs require the pinned and explicitly selected
   `stochtree` release. The NumPy backend remains an experimental comparator.
2. **`tvp`** ships as the Gaussian random-walk-coefficient trend; the
   BART-coefficient version awaits leaf-regression support.
3. **`multistep="direct"`** is not implemented (recursive is the plan's
   recommendation and default); the config field is reserved.
4. **Ensemble serialization** persists analysis artifacts (component draws, σ²,
   config); full ensemble round-tripping defers to `stochtree`'s `JSONSerializer`.
