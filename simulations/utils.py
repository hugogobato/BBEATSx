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
    return t, y, {"trend": trend, "seasonal": seasonal, "generic": ar}

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
    return t, y, {"trend": trend, "seasonal": seasonal, "generic": ar, "sigma": sig}

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
    return t, y, exog_data, {"trend": trend, "seasonal": seasonal, "generic": generic}

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
    return t, y, {"trend": trend, "seasonal": seasonal, "generic": noise}

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
    return t, y, exog_data, {"trend": trend, "seasonal": seasonal, "generic": generic}

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
    return t, y, exog_data, {"trend": trend, "seasonal": seasonal, "generic": generic}


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

def evaluate_component_fidelity(
    true_comps: Dict[str, np.ndarray], 
    pred_comps: Dict[str, np.ndarray], 
    offset: int = 0,
    level: float = 0.9
) -> Dict[str, Dict[str, float]]:
    """Evaluates component recovery RMSE and empirical coverage.
    true_comps: dict of true component arrays (length N)
    pred_comps: dict of predicted component draws (length N_eff, draws)
    offset: integer marking how many starting rows were dropped in prediction
    """
    results = {}
    for name in ["trend", "seasonal", "generic"]:
        if name in true_comps and name in pred_comps:
            pc = pred_comps[name]
            tc = true_comps[name][offset : offset + pc.shape[0]]
            
            # Point-estimate RMSE (posterior mean)
            mean_pred = pc.mean(axis=1)
            rmse = np.sqrt(np.mean((tc - mean_pred) ** 2))
            
            # Interval coverage
            a = (1.0 - level) / 2.0
            lo = np.quantile(pc, a, axis=1)
            hi = np.quantile(pc, 1.0 - a, axis=1)
            coverage = np.mean((tc >= lo) & (tc <= hi))
            
            results[name] = {"rmse": float(rmse), "coverage": float(coverage)}
    return results


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
