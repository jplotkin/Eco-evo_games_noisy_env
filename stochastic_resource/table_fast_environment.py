"""SI Table 3: stable interior equilibrium of the averaged replicator equation when
the resource is fast, with E[min(m,1) | x] from the exact quasi-stationary
distribution at frozen x; and E[min(m,1) | x] at K = 100."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
import model as M

PAIRS = [(3, 1, 2, 3), (3, 1, 2, -3), (-3, 1, 2, -3), (-3, 1, 2, 3)]
KS = [40, 100, 400, 1600, None]


@lru_cache(maxsize=None)
def mean_state(x, K):
    if K is None:
        return min(1.0, 1 - M.Q * M.ebar(x) / M.R)
    xs = np.full(3 * K + 40, x)
    u, _ = M.qsd_exact(*M.rates(K, xs))
    m = np.arange(1, len(u) + 1) / K
    return float(np.sum(u * np.minimum(m, 1.0)))


def Dbar(x, g, K):
    dL0, dH0, DL1, DH1 = g
    e = mean_state(float(x), K)
    return (1 - e) * (dL0 * x + dH0 * (1 - x)) - e * (DL1 * x + DH1 * (1 - x))


def stable_interior(g, K):
    xg = np.linspace(0, 1, 201)
    d = np.array([Dbar(x, g, K) for x in xg])
    roots = [brentq(Dbar, xg[i], xg[i + 1], args=(g, K), xtol=1e-12)
             for i in range(200) if d[i] > 0 > d[i + 1]]
    return roots


for g in PAIRS:
    vals = [stable_interior(g, K) for K in KS]
    print(g, "  ".join(f"K={K}: " + ",".join(f"{r:.3f}" for r in v) for K, v in zip(KS, vals)))
for x in (0.0, 0.5, 1.0):
    xs = np.full(340, x)
    u, _ = M.qsd_exact(*M.rates(100, xs))
    m = np.arange(1, len(u) + 1) / 100
    print(f"K=100 x={x}: E[min(m,1)]={np.sum(u*np.minimum(m,1)):.3f}  deterministic {1-M.Q*M.ebar(x)/M.R:.3f}"
          f"  Pr(m>1)={u[m > 1].sum():.3f}")
