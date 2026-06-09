"""Noise-variance recovery on a known-sigma DGP (plan §1.5 test c).

With a single dataset the sigma^2 posterior concentrates on the *realized*
in-sample noise level (which differs from the generating sigma0 by sampling
variability ~ sigma0/sqrt(2n)), so the calibrated check is that the posterior
credible interval covers the realized noise standard deviation, and that the
posterior mean is close to sigma0.
"""

import numpy as np

from bbeatsx import BBEATSx, make_config


def test_sigma_posterior_recovers_known_noise():
    n = 220
    sigma0 = 0.7
    rng = np.random.default_rng(7)
    t = np.arange(n)
    noise = rng.normal(0, sigma0, n)
    y = 1.0 + 0.04 * t + noise                      # linear trend + known noise
    realized_sd = float(np.std(noise))

    cfg = make_config(periods=[], lags=(), trend="linear", errors="homo",
                      num_gfr=5, num_burnin=200, num_mcmc=400, seed=0)
    m = BBEATSx(cfg).fit(y)

    # sigma^2 draws are on the standardized scale; map back to the data scale.
    s = m.sampler_
    sigma_orig = np.sqrt(np.array(s.sigma2_draws_)) * s.y_std_
    post_mean = sigma_orig.mean()
    lo, hi = np.quantile(sigma_orig, [0.05, 0.95])

    # posterior is calibrated to the realized noise level...
    assert lo < realized_sd < hi
    # ...and close to the generating sigma0.
    assert abs(post_mean - sigma0) < 0.12
