"""Stochastic simulations of the resource (Euler-Maruyama for the SDE of m) coupled to
the strategy frequency x, either at its stable rest point (timescale separation) or by
its ODE. As in the original simulations, a resource level that falls to zero or below
is reset to 1/1000 (reflecting boundary). Payoff levels for mean payoffs follow the
original notebook: T0 = 2, P0 = 1, R1 = 3, S1 = 2.
"""
import numpy as np
import numba
from scipy.integrate import trapezoid

LAM, MU, Q, EL, EH = 1.2, 0.7, 0.8, 0.1, 0.56
R = LAM - MU
T0, P0, R1, S1 = 2.0, 1.0, 3.0, 2.0
M_RESET = 1e-3


@numba.njit
def D(x, m, g0, g1, g2, g3):
    mh = min(m, 1.0)
    return (1 - mh) * (g0 * x + g1 * (1 - x)) - mh * (g2 * x + g3 * (1 - x))


@numba.njit
def rest_point(x, m, g0, g1, g2, g3):
    """Strategy frequency after the strategy dynamics relax at resource level m,
    starting from x. A boundary rest point is kept while it is stable."""
    a = D(1.0, m, g0, g1, g2, g3)
    b = D(0.0, m, g0, g1, g2, g3)
    if (x == 0.0 and b < 0) or (x == 1.0 and a > 0):
        return x
    if b > 0 > a:
        return b / (b - a)
    if b < 0 and not a > 0:
        return 0.0
    if a > 0 and not b < 0:
        return 1.0
    if a > 0 and b < 0:
        return 1.0 if a * x + b * (1 - x) > 0 else 0.0
    return x


@numba.njit
def mean_payoff(x, m, g0, g1, g2, g3):
    mh = min(m, 1.0)
    R0, S0 = T0 + g0, P0 + g1
    T1, P1 = R1 + g2, S1 + g3
    pL = (1 - mh) * (R0 * x + S0 * (1 - x)) + mh * (R1 * x + S1 * (1 - x))
    pH = (1 - mh) * (T0 * x + P0 * (1 - x)) + mh * (T1 * x + P1 * (1 - x))
    return x * pL + (1 - x) * pH


@numba.njit
def m_step(m, x, K, rate, dt):
    e = EL * x + EH * (1 - x)
    drift = rate * m * (R * (1 - m) - Q * e)
    var = rate * ((LAM + MU) * m + R * m * m) / K
    m = m + drift * dt + np.sqrt(var * dt) * np.random.normal()
    return m if m > 0 else M_RESET


@numba.njit
def separated(g0, g1, g2, g3, K, s, T, dt, m0, x0, every, seed):
    """Timescale separation: x sits at its rest point. Returns (t, x, m) sampled every
    `every` steps."""
    np.random.seed(seed)
    n = int(T / dt)
    out = np.empty((n // every + 1, 3))
    m, x = m0, rest_point(x0, m0, g0, g1, g2, g3)
    out[0] = 0.0, x, m
    k = 1
    for i in range(1, n + 1):
        m = m_step(m, x, K, s, dt)
        x = rest_point(x, m, g0, g1, g2, g3)
        if i % every == 0 and k < out.shape[0]:
            out[k] = i * dt, x, m
            k += 1
    return out[:k]


@numba.njit
def coupled(g0, g1, g2, g3, K, eps, s, T, dt, x0, m0, every, seed):
    """x by its ODE (Euler, in log-odds), m by its SDE; K <= 0 means no noise. Returns (t, x, m,
    mean payoff) sampled every `every` steps."""
    np.random.seed(seed)
    n = int(T / dt)
    out = np.empty((n // every + 1, 4))
    y = np.log(x0 / (1 - x0))                 # x in log-odds, so that x never rounds to 0 or 1
    x, m = x0, m0
    out[0] = 0.0, x, m, mean_payoff(x, m, g0, g1, g2, g3)
    k = 1
    for i in range(1, n + 1):
        dy = s * D(x, m, g0, g1, g2, g3) * dt
        if K > 0:
            m = m_step(m, x, K, s * eps, dt)
        else:
            e = EL * x + EH * (1 - x)
            m = m + s * eps * m * (R * (1 - m) - Q * e) * dt
        y = min(max(y + dy, -700.0), 700.0)
        x = 1.0 / (1.0 + np.exp(-y))
        if i % every == 0 and k < out.shape[0]:
            out[k] = i * dt, x, m, mean_payoff(x, m, g0, g1, g2, g3)
            k += 1
    return out[:k]


def stationary_m(g, K, m_grid, x_of_m, s=1.0, eps=1.0):
    """Diffusion approximation to the quasi-stationary density of m under timescale
    separation (Eq. 8 of the SI), normalized on the grid."""
    x = np.array([x_of_m(mm) for mm in m_grid])
    e = EL * x + EH * (1 - x)
    integrand = (R * (1 - m_grid) - Q * e) / ((LAM + MU) + R * m_grid)
    expo = 2 * K * np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(m_grid))))
    logf = expo - np.log((LAM + MU) * m_grid + R * m_grid ** 2)
    f = np.exp(logf - logf.max())
    return f / trapezoid(f, m_grid)
