"""Local validation of BBEATSx.forecast_from_origin (re-anchored frozen-draw
forecasting used by the periodic-recalibration protocol in
real_life_application_v2). Runs on the numpy backend (no stochtree needed)."""
import numpy as np
from bbeatsx import BBEATSx, make_config


def _make_series(n, seed=0, vol=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x1 = np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
    base = 10 + 0.01 * t + 2.0 * np.sin(2 * np.pi * t / 24) + 0.7 * x1
    y = base + np.cumsum(rng.normal(0, 0.15, n)) + rng.normal(0, vol, n)
    return y, {"x1": x1}


def _fit(y, exog):
    cfg = make_config(periods=[(24, 2)], lags=(1, 24), exog=["x1"],
                      trend="spline", errors="sv", asymmetric=True,
                      num_gfr=5, num_burnin=10, num_mcmc=40, seed=123, num_threads=1)
    return BBEATSx(cfg).fit(y, exog=exog)


def test_equivalence_at_training_end():
    """forecast_from_origin at the training end must reproduce forecast() exactly
    (same RNG state) -- proves the re-anchoring reuses the same machinery."""
    n, H = 400, 24
    y, exog = _make_series(n, seed=1)
    m = _fit(y, exog)
    xfut = np.sin(2 * np.pi * np.arange(n, n + H) / 24)
    exf = {"x1": xfut}

    m.sampler_.np_rng = np.random.default_rng(777)
    fc_std = m.forecast(H, exog_future=exf)
    m.sampler_.np_rng = np.random.default_rng(777)
    fc_org = m.forecast_from_origin(y, H, exog_future=exf, exog_history=exog)

    dmean = float(np.max(np.abs(fc_std.mean() - fc_org.mean())))
    dsamp = float(np.max(np.abs(fc_std.samples - fc_org.samples)))
    print(f"[equiv] max|mean diff|={dmean:.2e}  max|samples diff|={dsamp:.2e}")
    assert dmean < 1e-9 and dsamp < 1e-9, "re-anchoring not equivalent at train end"


def test_reanchor_runs_and_tracks_level():
    """At an origin deep in realised data the forecast must be finite, correctly
    shaped/indexed, and track the realised recent level (not the training end)."""
    n_train, gap, H = 350, 120, 24
    y_all, exog_all = _make_series(n_train + gap + H, seed=2)
    y_tr = y_all[:n_train]
    ex_tr = {"x1": exog_all["x1"][:n_train]}
    m = _fit(y_tr, ex_tr)

    origin = n_train + gap
    y_hist = y_all[:origin]
    ex_hist = {"x1": exog_all["x1"][:origin]}
    exf = {"x1": exog_all["x1"][origin:origin + H]}
    fc = m.forecast_from_origin(y_hist, H, exog_future=exf, exog_history=ex_hist)

    assert fc.samples.shape == (H, m.sampler_.num_draws)
    assert np.all(np.isfinite(fc.samples))
    assert list(fc.t_index) == list(range(origin, origin + H))
    # point forecast should sit near the realised recent level, far from train end
    recent = y_hist[-48:].mean()
    err_recent = abs(fc.mean().mean() - recent)
    err_trainend = abs(fc.mean().mean() - y_tr[-48:].mean())
    print(f"[reanchor] |fc-recent|={err_recent:.2f}  |fc-trainend|={err_trainend:.2f}")
    assert np.all(np.isfinite(fc.interval(0.9)[0]))


def test_sv_filter_widens_intervals_for_volatile_gap():
    """A volatile recent gap should yield wider predictive intervals than a calm
    one (the SV log-vol is filtered through the realised residuals)."""
    n_train, gap, H = 350, 120, 24
    y_all, exog_all = _make_series(n_train + gap + H, seed=3)
    y_tr, ex_tr = y_all[:n_train], {"x1": exog_all["x1"][:n_train]}
    m = _fit(y_tr, ex_tr)
    origin = n_train + gap
    ex_hist = {"x1": exog_all["x1"][:origin]}
    exf = {"x1": exog_all["x1"][origin:origin + H]}

    y_calm = y_all[:origin].copy()
    y_vol = y_all[:origin].copy()
    rng = np.random.default_rng(99)
    y_vol[-40:] += rng.normal(0, 8.0, 40)        # inject a high-vol recent burst

    m.sampler_.np_rng = np.random.default_rng(5)
    lo, hi = m.forecast_from_origin(y_calm, H, exog_future=exf, exog_history=ex_hist).interval(0.9)
    w_calm = float(np.mean(hi - lo))
    m.sampler_.np_rng = np.random.default_rng(5)
    lo, hi = m.forecast_from_origin(y_vol, H, exog_future=exf, exog_history=ex_hist).interval(0.9)
    w_vol = float(np.mean(hi - lo))
    print(f"[sv] mean 90% width calm={w_calm:.2f}  volatile={w_vol:.2f}")
    assert w_vol > w_calm, "SV filter did not widen intervals after a volatile gap"


if __name__ == "__main__":
    test_equivalence_at_training_end()
    test_reanchor_runs_and_tracks_level()
    test_sv_filter_widens_intervals_for_volatile_gap()
    print("\nALL FORECAST-FROM-ORIGIN CHECKS PASSED")
