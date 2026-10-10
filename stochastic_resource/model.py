"""Eco-evolutionary game with a finite, stochastic resource.

Strategy frequency x (low-impact players, harvest effort e_L; the rest use e_H).
Resource: M individuals, carrying capacity K, m = M/K.
    M -> M+1 at rate M * [lam - (1-theta) q ebar(x)]
    M -> M-1 at rate M * [mu + (lam-mu) M/K + theta q ebar(x)]
Rates are in units of 1/(s*eps). Under timescale separation x sits at a stable
rest point of dx/dt = x(1-x) D(x, m), with
    D(x, m) = (1-mh)[dL0 x + dH0 (1-x)] - mh[DL1 x + DH1 (1-x)],  mh = min(m, 1).
A game is the tuple g = (dL0, dH0, DL1, DH1).
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import logsumexp

LAM, MU, Q, EL, EH = 1.2, 0.7, 0.8, 0.1, 0.56
R = LAM - MU

GAMES = ["PD", "SD", "SH", "HG"]
NAMES = {"PD": "prisoner's dilemma", "SD": "snowdrift", "SH": "stag hunt", "HG": "harmony"}
SIGN_POOR = {"PD": (-1, -1), "SD": (-1, 1), "SH": (1, -1), "HG": (1, 1)}
SIGN_RICH = {"PD": (1, 1), "SD": (1, -1), "SH": (-1, 1), "HG": (-1, -1)}
MAG = (3, 1, 2, 3)


def game(poor, rich, mag=MAG):
    s0, s1 = SIGN_POOR[poor], SIGN_RICH[rich]
    return (s0[0] * mag[0], s0[1] * mag[1], s1[0] * mag[2], s1[1] * mag[3])


def ebar(x):
    return EL * x + EH * (1 - x)


def D(x, m, g):
    dL0, dH0, DL1, DH1 = g
    mh = min(m, 1.0)
    return (1 - mh) * (dL0 * x + dH0 * (1 - x)) - mh * (DL1 * x + DH1 * (1 - x))


def stable_points(m, g):
    a, b = D(1, m, g), D(0, m, g)
    pts = []
    if b < 0:
        pts.append(0.0)
    if a > 0:
        pts.append(1.0)
    if b > 0 > a:
        pts.append(b / (b - a))
    return pts


def follow(m_desc, g, x0):
    """Strategy frequency along decreasing resource levels. The population stays
    at a boundary rest point while it is stable and otherwise moves to the
    attractor of the strategy dynamics."""
    xs, x = np.empty(len(m_desc)), x0
    for i, m in enumerate(m_desc):
        pts = stable_points(m, g)
        if not (x in (0.0, 1.0) and x in pts):
            interior = [p for p in pts if 0 < p < 1]
            if interior:
                x = interior[0]
            elif len(pts) == 1:
                x = pts[0]
            else:
                x = 1.0 if D(x, m, g) > 0 else 0.0
        xs[i] = x
    return xs


def growth(m, x):
    """Per-capita growth rate of the resource, b_+ - b_-."""
    return R * (1 - m) - Q * ebar(x)


def bplus(m, x, theta=0.5):
    return LAM - (1 - theta) * Q * ebar(x)


def bminus(m, x, theta=0.5):
    return MU + R * m + theta * Q * ebar(x)


def attractors(g):
    """Stable equilibria (m*, x*) of the deterministic system under timescale
    separation, sorted by resource level."""
    out = []
    m0 = 1 - Q * EH / R                       # all high impact
    if D(0, m0, g) < 0:
        out.append((m0, 0.0))
    m1 = 1 - Q * EL / R                       # all low impact
    if D(1, m1, g) > 0:
        out.append((m1, 1.0))
    m = np.linspace(1e-6, 1.0, 200001)        # interior rest point of the strategy dynamics
    a = np.array([D(1, mm, g) for mm in m])
    b = np.array([D(0, mm, g) for mm in m])
    ok = (b > 0) & (a < 0)
    x2 = np.where(ok, b / np.where(ok, b - a, 1), np.nan)
    f = np.where(ok, growth(m, x2), np.nan)
    for i in range(len(m) - 1):
        if ok[i] and ok[i + 1] and f[i] > 0 >= f[i + 1]:
            mm = m[i] - f[i] * (m[i + 1] - m[i]) / (f[i + 1] - f[i])
            out.append((mm, float(x2[i])))
    return sorted(out)


def path(g, m_star, x_star, n=40001):
    """Resource grid from m* down to 0 (returned in increasing order) and the
    strategy frequency the population occupies along it."""
    m = np.linspace(m_star, 1e-7, n)
    x = follow(m, g, x_star)
    return m[::-1], x[::-1]


def barrier(g, m_star, x_star, theta=0.5, n=40001):
    """Collapse barrier Delta V = max over u <= v <= m* of the integral from u to v
    of ln(b_+/b_-), and the function V(m) - V(m*) on [0, m*]."""
    m, x = path(g, m_star, x_star, n)
    lg = np.log(bplus(m, x, theta) / bminus(m, x, theta))
    phi = np.concatenate(([0.0], np.cumsum(0.5 * (lg[1:] + lg[:-1]) * np.diff(m))))
    best = np.maximum.accumulate(phi[::-1])[::-1]          # max of phi over v >= u
    dV = float(np.max(best - phi))
    return dV, m, x, phi[-1] - phi


def rates(K, xs, theta=0.5, iota=0.0):
    """Birth and death rates for M = 1..len(xs), with x = xs[M-1]."""
    M = np.arange(1, len(xs) + 1, dtype=float)
    m = M / K
    birth = M * bplus(m, xs, theta) + iota * K
    death = M * bminus(m, xs, theta)
    return birth, death


def chain_x(g, K, x_star, Mmax=None):
    """Strategy frequency at M = 1..Mmax, followed down from the top state."""
    if Mmax is None:
        Mmax = int(3 * K) + 40
    m = np.arange(1, Mmax + 1) / K
    return follow(m[::-1], g, x_star)[::-1]


def log_product(birth, death):
    """log of pi_M = prod_{j=2}^{M} birth_{j-1}/death_j for M = 1..N (pi_1 = 1)."""
    return np.concatenate(([0.0], np.cumsum(np.log(birth[:-1]) - np.log(death[1:]))))


def mean_time_to_zero(birth, death, n):
    """Exact mean time to reach M = 0 from M = n:
    T(n) = sum_{k<=n} (1/(death_k pi_k)) sum_{j>=k} pi_j."""
    lp = log_product(birth, death)
    tail = np.logaddexp.accumulate(lp[::-1])[::-1]
    return float(np.exp(logsumexp(tail[:n] - lp[:n] - np.log(death[:n]))))


def qsd_product(birth, death):
    lp = log_product(birth, death)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def qsd_exact(birth, death):
    """Quasi-stationary distribution: left eigenvector, for the eigenvalue of
    largest real part, of the generator restricted to M >= 1 (births switched
    off in the top state). Returns (distribution, decay rate)."""
    b = birth.copy()
    b[-1] = 0.0
    diag = -(b + death)
    off = np.sqrt(birth[:-1] * death[1:])                 # symmetrized tridiagonal
    w, v = eigh_tridiagonal(diag, off, select="i", select_range=(len(diag) - 1, len(diag) - 1))
    lp = log_product(birth, death)
    lu = np.log(np.abs(v[:, 0]) + 1e-300) + 0.5 * lp
    u = np.exp(lu - lu.max())
    return u / u.sum(), -float(w[0])


def variance_formula(m_star, slope, theta=0.5):
    """K Var(m) to leading order in 1/K; slope = dx/dm at m*."""
    num = theta * LAM + (1 - theta) * (MU + R * m_star)
    return num / (R - Q * (EH - EL) * slope)


def interior_slope(m, g):
    """dx/dm of the interior rest point of the strategy dynamics."""
    dL0, dH0, DL1, DH1 = g
    den = (dH0 - dL0) - m * ((dH0 + DH1) - (dL0 + DL1))
    return (dL0 * DH1 - dH0 * DL1) / den ** 2
