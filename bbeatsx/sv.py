"""Stochastic-volatility error model for BBEATSx (plan §0.2, §1.2 step 4).

The model residual ``eps_t = y_t - F_tr - F_se - F_ge`` is given a stochastic
volatility law

    eps_t = exp(h_t / 2) z_t,  z_t ~ N(0, 1),
    h_t   = mu + phi (h_{t-1} - mu) + sigma_h nu_t,  nu_t ~ N(0, 1).

Conditioning on ``eps_t``, ``log(eps_t^2) = h_t + log(z_t^2)`` and ``log(z_t^2)``
(a ``log chi^2_1`` variate) is approximated by the **Omori, Chib, Shephard &
Nakajima (2007) 10-component Gaussian mixture**.  Given the mixture indicators the
state space is linear-Gaussian and ``h_{1:T}`` is drawn by forward-filter /
backward-sample (the Kastner--Fruehwirth-Schnatter "all-without-a-loop" idea, here
in its plain FFBS form).  ``phi`` and ``sigma_h`` are held at their prior values
in this Phase-1 implementation; the volatility level ``mu`` is updated by its
Gaussian full conditional.

**Exact-target correction (theory flag F-2.4-8, ``exact=True`` default).**  The
mean blocks are exact conditionals of the *exact* SV posterior (Gaussian
observation density ``phi(eps_t; 0, e^{h_t})``), while the raw FFBS h-block is an
exact conditional of the *Omori-augmented* posterior; composed, the sweep is
invariant for neither (theory/section.tex, prop:svtarget).  Treating the FFBS
draw as a Metropolis--Hastings proposal and accepting with probability

    alpha = min{1, exp[sum_t l_t(h'_t) - l_t(h_t)]},
    l_t(v) = log phi(eps_t; 0, e^v) - log q(u_t - v),

where ``q`` is the mixture density and ``u_t = log(eps_t^2 + c)``, restores exact
invariance for the exact SV posterior at O(n) cost per sweep (thm:svfix); the
mixture indicators cancel from the ratio and the offset ``c`` is corrected
automatically.  Measured acceptance is ~0.98 at n=100 down to ~0.92 at n=3200.

The per-observation variance ``sigma_t^2 = exp(h_t)`` is what the forest and trend
blocks condition on (via precision weights), and it is propagated forward through
its AR(1) law when forecasting (plan §0.5 / §1.3).
"""

from __future__ import annotations

import numpy as np

# Omori et al. (2007) 10-component mixture approximation to log(chi^2_1).
# Columns: probability p_i, mean m_i, variance v_i^2. The probability-weighted
# mean is ~= -1.2704, matching E[log chi^2_1].
_OMORI_P = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
                     0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
_OMORI_M = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
                     -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
_OMORI_V2 = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
                      0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
_OMORI_V = np.sqrt(_OMORI_V2)
_LOG_P = np.log(_OMORI_P)


class SVSampler:
    """Sweep-wise sampler for the AR(1) log-variance path ``h_{1:T}``."""

    def __init__(self, n: int, phi: float = 0.95, sigma_h: float = 0.15,
                 mu_prior_var: float = 10.0, init_h: float = 0.0,
                 exact: bool = True) -> None:
        self.n = int(n)
        self.phi = float(phi)
        self.sigma_h = float(sigma_h)
        self.sigma_h2 = self.sigma_h ** 2
        self.mu_prior_var = float(mu_prior_var)
        self.mu = float(init_h)
        self.h = np.full(self.n, float(init_h))
        self._initialized = False
        # F-2.4-8: MH-correct the FFBS proposal so the h-block targets the
        # exact SV posterior rather than the Omori-augmented one (thm:svfix).
        self.exact = bool(exact)
        self.mh_proposals = 0
        self.mh_accepts = 0

    @property
    def mh_acceptance_rate(self) -> float:
        """Acceptance rate of the exact-target MH step (1.0 if never proposed)."""
        return self.mh_accepts / self.mh_proposals if self.mh_proposals else 1.0

    @staticmethod
    def _log_ell(eps: np.ndarray, u: np.ndarray, h: np.ndarray) -> float:
        """``sum_t l_t(h_t)`` with ``l_t(v) = log phi(eps_t;0,e^v) - log q(u_t-v)``.

        The ``-log(2 pi)/2`` constants cancel between the two terms' differences
        in the MH ratio but are kept so the value is the exact log-weight of
        KSC (1998) section 3.4 up to their additive constant.
        """
        # log N(eps; 0, e^h)
        log_phi = -0.5 * np.log(2.0 * np.pi) - 0.5 * h - 0.5 * eps ** 2 * np.exp(-h)
        # log q(u - h): 10-component Gaussian mixture, via logsumexp
        z = u[:, None] - h[:, None] - _OMORI_M[None, :]
        comp = (_LOG_P[None, :] - np.log(_OMORI_V)[None, :]
                - 0.5 * np.log(2.0 * np.pi) - 0.5 * z ** 2 / _OMORI_V2[None, :])
        cmax = comp.max(axis=1)
        log_q = cmax + np.log(np.exp(comp - cmax[:, None]).sum(axis=1))
        return float(np.sum(log_phi - log_q))

    # ---------------------------------------------------------------- one sweep
    def step(self, eps: np.ndarray, rng) -> np.ndarray:
        """Draw ``h_{1:T}`` and ``mu`` given the current residual; return sigma_t^2."""
        gen = rng.rng if hasattr(rng, "rng") else rng
        eps = np.asarray(eps, dtype=float).ravel()
        if not self._initialized:
            # Initialise the level from the residual scale.
            self.mu = float(np.log(np.var(eps) + 1e-8))
            self.h[:] = self.mu
            self._initialized = True

        u = np.log(eps ** 2 + 1e-10)               # log(eps_t^2)

        # 1. mixture indicators s_t | h_t, u_t
        # log weight_{t,i} = log p_i - log v_i - (u_t - h_t - m_i)^2 / (2 v_i^2)
        resid = u[:, None] - self.h[:, None] - _OMORI_M[None, :]
        logw = (_LOG_P[None, :] - np.log(_OMORI_V)[None, :]
                - 0.5 * resid ** 2 / _OMORI_V2[None, :])
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        cdf = np.cumsum(w, axis=1)
        draws = gen.random(self.n)[:, None]
        s = (draws > cdf).sum(axis=1)
        s = np.clip(s, 0, 9)

        m_s = _OMORI_M[s]
        V_s = _OMORI_V2[s]
        ytil = u - m_s                              # observation: ytil_t = h_t + N(0,V_s)

        # 2. FFBS for the AR(1) state g_t = h_t - mu.  With exact=True the FFBS
        # draw is a MH *proposal* for the exact SV posterior (thm:svfix): the
        # indicators cancel from the ratio, which is exp[sum l(h') - l(h)].
        h_prop = self._ffbs(ytil, V_s, gen) + self.mu
        if self.exact:
            log_ratio = (self._log_ell(eps, u, h_prop)
                         - self._log_ell(eps, u, self.h))
            self.mh_proposals += 1
            if np.log(gen.random()) < log_ratio:
                self.h = h_prop
                self.mh_accepts += 1
            # else: retain the current path h (reject)
        else:
            self.h = h_prop

        # 3. update mu | h path (Gaussian conditional; exact for the SV
        # posterior since the observation density does not involve mu)
        self._update_mu(gen)

        return np.exp(self.h)

    def _ffbs(self, ytil: np.ndarray, V: np.ndarray, gen) -> np.ndarray:
        phi, sh2 = self.phi, self.sigma_h2
        mu = self.mu
        n = self.n
        m = np.zeros(n)
        C = np.zeros(n)
        # stationary prior on g_0
        a = 0.0
        P = sh2 / (1.0 - phi ** 2)
        for t in range(n):
            if t > 0:
                a = phi * m[t - 1]
                P = phi ** 2 * C[t - 1] + sh2
            Q = P + V[t]
            gain = P / Q
            m[t] = a + gain * ((ytil[t] - mu) - a)
            C[t] = P * (1.0 - gain)

        g = np.zeros(n)
        g[-1] = gen.normal(m[-1], np.sqrt(max(C[-1], 1e-12)))
        for t in range(n - 2, -1, -1):
            a_next = phi * m[t]
            R_next = phi ** 2 * C[t] + sh2
            B = C[t] * phi / R_next
            mean = m[t] + B * (g[t + 1] - a_next)
            var = C[t] - B ** 2 * R_next
            g[t] = gen.normal(mean, np.sqrt(max(var, 1e-12)))
        return g

    def _update_mu(self, gen) -> None:
        phi, sh2 = self.phi, self.sigma_h2
        h = self.h
        n = self.n
        # contributions: h_0 ~ N(mu, sh2/(1-phi^2)); h_t ~ N(mu + phi(h_{t-1}-mu), sh2)
        prec = 1.0 / self.mu_prior_var
        rhs = 0.0
        p0 = (1.0 - phi ** 2) / sh2
        prec += p0
        rhs += p0 * h[0]
        c = 1.0 - phi
        if n > 1:
            pc = c ** 2 / sh2
            prec += (n - 1) * pc
            rhs += (c / sh2) * np.sum(h[1:] - phi * h[:-1])
        var = 1.0 / prec
        self.mu = gen.normal(var * rhs, np.sqrt(var))

    # --------------------------------------------------------------- forecasting
    def forecast_path(self, h_last: float, horizon: int, gen,
                      mu: float | None = None) -> np.ndarray:
        """Propagate one volatility path forward ``horizon`` steps (returns sigma^2).

        ``mu`` is the volatility level of the *retained draw* being propagated
        (fix F-2.4-1: the predictive must carry the posterior spread of ``mu``,
        not the final sweep's value).  Falls back to the current ``self.mu`` when
        omitted, which reproduces the pre-fix behaviour.
        """
        level = self.mu if mu is None else float(mu)
        out = np.empty(horizon)
        h = h_last
        for k in range(horizon):
            h = level + self.phi * (h - level) + self.sigma_h * gen.normal()
            out[k] = np.exp(h)
        return out
