import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Dict, Tuple, Any

# =====================================================================
# 1. Controlled Data Generating Processes (DGPs) - Plan §3.1
# =====================================================================

def generate_dgp1(n: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """DGP 1: Linear Trend + Single Seasonality + AR(1) + Homoscedastic Noise
    Standard baseline to test regular additive structure.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    # Linear trend
    trend = 1.0 + 0.03 * t
    # Single-period seasonality (frequency 12, e.g., monthly)
    seasonal = 1.5 * np.sin(2 * np.pi * t / 12)
    # AR(1) process
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.5 * ar[i-1] + rng.normal(0, 0.4)
        
    y = trend + seasonal + ar
    return t, y, {"trend": trend, "seasonal": seasonal, "generic": ar,
                  # Metadata for the *predictable* generic estimand (see
                  # `generic_predictable_in/out`): the generic block can only
                  # recover E[g_t | past], never the innovation on top of it.
                  "generic_ar": ar, "generic_exog": np.zeros(n),
                  "generic_ar_coefs": np.array([0.5]), "generic_innov_sd": 0.4}

def generate_dgp2(n: int = 250, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """DGP 2: Nonlinear Trend + Multi-Period Seasonality + AR(2) + Volatility Clustered (SV) Noise
    Highly complex setup to evaluate the SV error model and non-linear trend.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    # Nonlinear trend (quadratic + slow cycle)
    trend = 2.0 + 0.03 * t - 0.00015 * (t ** 2) + 0.5 * np.sin(2 * np.pi * t / 80)
    # Multi-period seasonality (daily + weekly cycle on daily data, e.g. 7 and 12)
    seasonal = 1.2 * np.sin(2 * np.pi * t / 12) + 0.8 * np.sin(2 * np.pi * t / 7)
    
    # Stochastic Volatility process for the error:
    # h_t = c + rho * (h_{t-1} - c) + sigma_h * nu_t
    h = np.zeros(n)
    h[0] = -1.2
    for i in range(1, n):
        h[i] = -1.2 + 0.9 * (h[i-1] + 1.2) + rng.normal(0, 0.15)
    sig = np.exp(h / 2.0)
    eps = rng.normal(0, 1, n) * sig
    
    # AR(2) generic block
    ar = np.zeros(n)
    for i in range(2, n):
        ar[i] = 0.5 * ar[i-1] - 0.25 * ar[i-2] + eps[i]
        
    y = trend + seasonal + ar
    return t, y, {"trend": trend, "seasonal": seasonal, "generic": ar, "sigma": sig,
                  "generic_ar": ar, "generic_exog": np.zeros(n),
                  "generic_ar_coefs": np.array([0.5, -0.25]),
                  "generic_innov_sd": sig}

def generate_dgp3(n: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """DGP 3: Structural Break Regime + Exogenous Covariate + Homoscedastic Noise
    Tests block recovery with exogenous inputs and sudden shifts.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    # Structural break in trend (level shift at midpoint)
    trend = np.zeros(n)
    trend[:n//2] = 1.0
    trend[n//2:] = 4.0
    
    # Seasonality
    seasonal = 1.2 * np.sin(2 * np.pi * t / 12)
    
    # Exogenous covariate (e.g., simulated temperature or marketing spend)
    x = rng.normal(0, 1.0, n)
    exog_data = {"x": x}
    
    # Exogenous effect is non-linear (sine-wave)
    exog_effect = 1.5 * np.sin(x)
    
    # AR(1) process
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.4 * ar[i-1] + rng.normal(0, 0.5)
        
    generic = exog_effect + ar
    y = trend + seasonal + generic
    return t, y, exog_data, {"trend": trend, "seasonal": seasonal, "generic": generic,
                             "generic_ar": ar, "generic_exog": exog_effect,
                             "generic_ar_coefs": np.array([0.4]),
                             "generic_innov_sd": 0.5}

def generate_dgp4(n: int = 80, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """DGP 4: Short + Noisy
    Tests BBEATSx regularized prior behavior in small-n, high-noise regimes.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    trend = 0.5 + 0.02 * t
    seasonal = 1.0 * np.cos(2 * np.pi * t / 4)
    # High noise-to-signal ratio
    noise = rng.normal(0, 1.2, n)
    
    y = trend + seasonal + noise
    return t, y, {"trend": trend, "seasonal": seasonal, "generic": noise,
                  # The "generic" component here is *pure white noise*: it has no
                  # predictable part at all, and the DGP4 configuration gives the
                  # model no generic features, so this block is reported N/A.
                  "generic_ar": noise, "generic_exog": np.zeros(n),
                  "generic_ar_coefs": np.array([]), "generic_innov_sd": 1.2}

def generate_dgp5(n: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """DGP 5: Linear Trend + Seasonality + Linear Exogenous Covariate + Homoscedastic Noise
    Tests recovery of a purely linear relationship on exogenous features.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    trend = 1.0 + 0.03 * t
    seasonal = 1.5 * np.sin(2 * np.pi * t / 12)
    
    # Exogenous covariate with a linear effect
    x = rng.normal(0, 1.0, n)
    exog_data = {"x": x}
    exog_effect = 2.0 * x
    
    # AR(1) process
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.5 * ar[i-1] + rng.normal(0, 0.4)
        
    generic = exog_effect + ar
    y = trend + seasonal + generic
    return t, y, exog_data, {"trend": trend, "seasonal": seasonal, "generic": generic,
                             "generic_ar": ar, "generic_exog": exog_effect,
                             "generic_ar_coefs": np.array([0.5]),
                             "generic_innov_sd": 0.4}

def generate_dgp6(n: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """DGP 6: Linear Trend + Seasonality + Nonlinear Exogenous Covariate + Homoscedastic Noise
    Tests recovery of a complex non-linear relationship on exogenous features.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    
    trend = 1.0 + 0.03 * t
    seasonal = 1.5 * np.sin(2 * np.pi * t / 12)
    
    # Exogenous covariate with a non-linear effect: 1.5 * sin(2*x) + 0.5 * x^2
    x = rng.uniform(-2.0, 2.0, n)
    exog_data = {"x": x}
    exog_effect = 1.5 * np.sin(2.0 * x) + 0.5 * (x ** 2)
    
    # AR(1) process
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.5 * ar[i-1] + rng.normal(0, 0.4)
        
    generic = exog_effect + ar
    y = trend + seasonal + generic
    return t, y, exog_data, {"trend": trend, "seasonal": seasonal, "generic": generic,
                             "generic_ar": ar, "generic_exog": exog_effect,
                             "generic_ar_coefs": np.array([0.5]),
                             "generic_innov_sd": 0.4}


# =====================================================================
# 2. Performance Metrics - Plan §3.3
# =====================================================================

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
    """Computes CRPS for a sample-based predictive distribution.
    actual: (H,), samples: (H, S)
    """
    H, S = samples.shape
    term1 = np.mean(np.abs(samples - actual[:, None]), axis=1)
    samples_sorted = np.sort(samples, axis=1)
    coef = 2.0 * np.arange(1, S + 1) - S - 1
    term2 = np.sum(coef[None, :] * samples_sorted, axis=1) / (S ** 2)
    return float(np.mean(term1 - term2))

def compute_pinball_loss(actual: np.ndarray, samples: np.ndarray, quantiles: np.ndarray = None) -> float:
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
    score = (hi - lo) + (2.0 / alpha) * (lo - actual) * (actual < lo) + (2.0 / alpha) * (actual - hi) * (actual > hi)
    return float(np.mean(score))

def evaluate_forecast(actual: np.ndarray, fc_result, y_train: np.ndarray, level: float = 0.9) -> Dict[str, float]:
    """Helper that evaluates a full battery of point and probabilistic forecast metrics."""
    samples = fc_result.samples
    mean_pred = fc_result.mean()
    return {
        "rmse": compute_rmse(actual, mean_pred),
        "smape": compute_smape(actual, mean_pred),
        "mase": compute_mase(actual, mean_pred, y_train),
        "crps": compute_crps(actual, samples),
        "pinball": compute_pinball_loss(actual, samples),
        "coverage": compute_coverage(actual, samples, level=level),
        "interval_score": compute_interval_score(actual, samples, level=level)
    }


# =====================================================================
# 3. Component-level and Decomposition UQ Metrics - Plan §3.2
# =====================================================================

def _band_metrics(true: np.ndarray, draws: np.ndarray, level: float) -> Dict[str, float]:
    """Component metrics in both the *raw* and the *identification-invariant* gauge.

    An additive decomposition determines the sum of the components but not how a
    constant is split among them, so a component's raw error confounds two very
    different things: whether the model recovered the component's *shape*, and
    which of the observationally equivalent level splits the model and the DGP
    each happen to use.  We therefore report both, plus the exact decomposition
    that links them.

    With ``d_t = mean_s pred_t^(s) - true_t``, ``offset = mean_t d_t`` and
    ``shape_rmse^2 = mean_t (d_t - offset)^2``:

        ``rmse^2 = offset^2 + shape_rmse^2``   (exact)

    The ``shape_*`` metrics recenter *both* the truth and every posterior draw
    over the evaluation window, so they are invariant to the level gauge and
    measure recovery of the component's actual movement.
    """
    a = (1.0 - level) / 2.0
    mean_pred = draws.mean(axis=1)
    d = mean_pred - true
    offset = float(np.mean(d))

    lo, hi = np.quantile(draws, a, axis=1), np.quantile(draws, 1.0 - a, axis=1)
    rmse = float(np.sqrt(np.mean(d ** 2)))
    coverage = float(np.mean((true >= lo) & (true <= hi)))

    # Gauge-invariant view: recenter truth and draws on their own window means.
    true_c = true - true.mean()
    draws_c = draws - mean_pred.mean()
    lo_c, hi_c = np.quantile(draws_c, a, axis=1), np.quantile(draws_c, 1.0 - a, axis=1)
    shape_rmse = float(np.sqrt(np.mean((d - offset) ** 2)))
    shape_coverage = float(np.mean((true_c >= lo_c) & (true_c <= hi_c)))

    return {
        "rmse": rmse, "coverage": coverage, "offset": offset,
        "shape_rmse": shape_rmse, "shape_coverage": shape_coverage,
        "width": float(np.mean(hi - lo)), "true_sd": float(np.std(true)),
    }


_NA_METRICS = {k: float("nan") for k in
               ("rmse", "coverage", "offset", "shape_rmse", "shape_coverage",
                "width", "true_sd")}


def evaluate_component_fidelity(
    true_comps: Dict[str, np.ndarray],
    pred_comps: Dict[str, np.ndarray],
    offset: int = 0,
    level: float = 0.9,
    available: Dict[str, bool] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate component recovery: raw, gauge-invariant, and their decomposition.

    Parameters
    ----------
    true_comps : dict
        True component arrays (length ``N``) from the DGP generator.
    pred_comps : dict
        Posterior component draws, ``(N_eff, draws)``.
    offset : int
        Number of leading rows dropped in prediction (the model's lag burn-in).
    level : float
        Nominal central-band level (the study uses 0.95).
    available : dict, optional
        Per-component flag; a component the *configuration* does not model at all
        (e.g. the DGP4 generic block, where ``lags=()`` and there is no exogenous
        input, so the block does not exist) is reported as NaN rather than as a
        zero-width band scoring 0.0 coverage against pure noise.
    """
    available = available or {}
    results = {}
    for name in ["trend", "seasonal", "generic"]:
        if name not in true_comps or name not in pred_comps:
            continue
        if not available.get(name, True):
            results[name] = dict(_NA_METRICS)
            continue
        pc = np.asarray(pred_comps[name], dtype=float)
        tc = np.asarray(true_comps[name], dtype=float)[offset: offset + pc.shape[0]]
        results[name] = _band_metrics(tc, pc, level)
    return results


# ---------------------------------------------------------------------
# The *predictable* generic estimand
# ---------------------------------------------------------------------
# Several DGPs define their "generic" component as a realised noise path
# (e.g. DGP1's ar_t = 0.5 ar_{t-1} + eps_t).  Scoring a posterior band against
# that path asks the model to have predicted eps_t, which is by construction
# impossible: no estimator, however good, can beat an RMSE floor of sd(eps).
# The estimable object is the conditional mean E[g_t | past, x_t], which is what
# the generic block actually targets; the innovation belongs to the observation
# noise and is carried by the *predictive* interval, not the component band.

def _companion(coefs: np.ndarray) -> np.ndarray:
    """Companion matrix of an AR(p) with coefficients ``coefs``."""
    p = len(coefs)
    A = np.zeros((p, p))
    A[0, :] = coefs
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    return A


def generic_predictable_in(comps: Dict[str, np.ndarray]) -> np.ndarray:
    """One-step conditional mean ``E[g_t | F_{t-1}, x_t]`` over the full series.

    Equals the (known) exogenous effect at ``t`` plus the AR forecast built from
    realised lags.  The first ``p`` entries are set to NaN and are never used:
    the model drops at least that many rows as lag burn-in.
    """
    ar = np.asarray(comps["generic_ar"], dtype=float)
    exog = np.asarray(comps["generic_exog"], dtype=float)
    coefs = np.asarray(comps["generic_ar_coefs"], dtype=float)
    n, p = ar.shape[0], len(coefs)
    psi = exog.astype(float).copy()
    for j, phi in enumerate(coefs, start=1):
        psi[j:] += phi * ar[:-j] if j < n else 0.0
    if p:
        psi[:p] = np.nan
    return psi


def generic_predictable_out(comps: Dict[str, np.ndarray], n_train: int,
                            H: int) -> np.ndarray:
    """``h``-step conditional mean ``E[g_{T+h} | F_T, x_{T+h}]`` for ``h = 1..H``.

    ``T = n_train - 1`` is the forecast origin.  The exogenous part is known into
    the future (it is supplied to the forecaster); the autoregressive part decays
    as the ``h``-th power of the companion matrix.
    """
    ar = np.asarray(comps["generic_ar"], dtype=float)
    exog = np.asarray(comps["generic_exog"], dtype=float)
    coefs = np.asarray(comps["generic_ar_coefs"], dtype=float)
    p = len(coefs)
    out = exog[n_train: n_train + H].astype(float).copy()
    if p == 0:
        return out                      # white noise: nothing is predictable
    A = _companion(coefs)
    state = ar[n_train - p: n_train][::-1]      # (ar_T, ar_{T-1}, ...)
    Ah = np.eye(p)
    for h in range(H):
        Ah = Ah @ A
        out[h] += float((Ah @ state)[0])
    return out


def generic_rmse_floor(comps: Dict[str, np.ndarray], n_train: int = None,
                       H: int = None, window: slice = None) -> float:
    """Irreducible RMSE of *any* estimator of the generic component.

    In sample this is the innovation sd (the one-step-unpredictable part).  Out of
    sample it also accumulates over the horizon: for an AR(p) the ``h``-step error
    variance is ``sigma^2 * sum_{i<h} psi_i^2`` with ``psi_i = [A^i]_{11}``.  The
    returned value is the root-mean-square of those per-step floors, directly
    comparable with the reported RMSE.
    """
    sd = comps["generic_innov_sd"]
    coefs = np.asarray(comps["generic_ar_coefs"], dtype=float)
    if H is None:                                       # in-sample
        sd_arr = np.asarray(sd, dtype=float)
        if sd_arr.ndim == 0:
            return float(sd_arr)
        return float(np.sqrt(np.mean(sd_arr[window] ** 2))) if window is not None \
            else float(np.sqrt(np.mean(sd_arr ** 2)))
    sd_arr = np.asarray(sd, dtype=float)
    sigma2 = (float(sd_arr) ** 2 if sd_arr.ndim == 0
              else float(np.mean(sd_arr[n_train: n_train + H] ** 2)))
    p = len(coefs)
    if p == 0:
        return float(np.sqrt(sigma2))
    A = _companion(coefs)
    Ah, var_h = np.eye(p), []
    cum = 0.0
    for _ in range(H):
        cum += float(Ah[0, 0]) ** 2
        var_h.append(sigma2 * cum)
        Ah = Ah @ A
    return float(np.sqrt(np.mean(var_h)))


# =====================================================================
# 4. MCMC Convergence Diagnostics - Plan §3.5
# =====================================================================

def compute_autocorr(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if n <= lag:
        return 0.0
    x_mean = np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return 0.0
    return float(np.mean((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / var)

def compute_ess(x: np.ndarray) -> float:
    """Computes the Effective Sample Size (ESS) for a 1D trace."""
    S = len(x)
    if S <= 2:
        return float(S)
    
    autocorrs = []
    for lag in range(1, min(S - 1, 100)):
        rho = compute_autocorr(x, lag)
        autocorrs.append(rho)
        
    sum_rho = 0.0
    for i in range(0, len(autocorrs) - 1, 2):
        pair_sum = autocorrs[i] + autocorrs[i+1]
        if pair_sum < 0:
            break
        sum_rho += pair_sum
        
    ess = S / (1.0 + 2.0 * sum_rho)
    return float(min(S, max(1.0, ess)))

def compute_gelman_rubin(chains: np.ndarray) -> float:
    """Computes the Gelman-Rubin R-hat statistic across multiple chains.
    chains: numpy array of shape (num_chains, num_draws)
    """
    C, S = chains.shape
    if C < 2:
        raise ValueError("Gelman-Rubin diagnostic requires at least 2 chains")
        
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = (S / (C - 1)) * np.sum((chain_means - overall_mean) ** 2)
    
    # Within-chain variance
    W = (1.0 / (C * (S - 1))) * np.sum((chains - chain_means[:, None]) ** 2)
    
    # Marginal posterior variance
    var_est = ((S - 1) / S) * W + (1.0 / S) * B
    
    if W < 1e-12:
        return 1.0
    r_hat = np.sqrt(var_est / W)
    return float(r_hat)
