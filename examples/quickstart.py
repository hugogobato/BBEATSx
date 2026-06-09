"""BBEATSx quickstart (Phase 1).

Fits BBEATSx on a synthetic trend + seasonal + AR series, forecasts with
per-component credible bands, and prints the interpretability summaries. Run:

    BBEATSX_BACKEND=numpy PYTHONPATH=. python examples/quickstart.py

Install `stochtree` (and drop the env var) to use the production sampler. Pass
``--plot`` to render the decomposition (needs matplotlib).
"""

import argparse

import numpy as np

from bbeatsx import BBEATSx, make_config, BACKEND


def make_data(n=220, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 1.0 + 0.04 * t
    seasonal = 1.5 * np.sin(2 * np.pi * t / 12)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal(0, 0.5)
    return t, trend + seasonal + e


def main(plot: bool) -> None:
    print(f"BBEATSx backend: {BACKEND}")
    t, y = make_data()
    n_train, horizon = 200, 20
    y_train, y_test = y[:n_train], y[n_train:n_train + horizon]

    cfg = make_config(periods=[(12, 3)], lags=(1, 2), trend="spline",
                      errors="homo", num_gfr=10, num_burnin=150, num_mcmc=300,
                      seed=0)
    model = BBEATSx(cfg).fit(y_train)
    print(f"retained draws: {model.sampler_.num_draws}")
    print(f"backfitting invariant error: "
          f"{model.sampler_.backfitting_residual_error():.2e}")

    fc = model.forecast(horizon)
    lo, hi = fc.interval(0.9)
    rmse = np.sqrt(np.mean((fc.mean() - y_test) ** 2))
    cov = np.mean((y_test >= lo) & (y_test <= hi))
    print(f"\nforecast RMSE vs held-out: {rmse:.3f}")
    print(f"90% interval coverage on held-out horizon: {cov:.2f}")

    tlo, thi = fc.component_interval("trend", 0.9)
    print(f"\ntrend component at h=1: {fc.component_mean('trend')[0]:.2f} "
          f"[{tlo[0]:.2f}, {thi[0]:.2f}]")

    imp = model.split_importance("generic")
    print("\ngeneric-block split-frequency importance (posterior mean):")
    for name, m, p in zip(imp.names, imp.mean, imp.inclusion_prob):
        print(f"  {name:8s}  mean splits={m:6.2f}  P(used)={p:.2f}")

    if plot:
        import matplotlib.pyplot as plt
        from bbeatsx.interpret import plot_decomposition
        plot_decomposition(model.sampler_)
        plt.tight_layout()
        plt.savefig("decomposition.png", dpi=110)
        print("\nsaved decomposition.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    main(ap.parse_args().plot)
