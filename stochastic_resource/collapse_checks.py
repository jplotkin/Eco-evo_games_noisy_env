"""Numbers in SI Sections 1.6, 1.7 and 1.9: product formula versus the exact
quasi-stationary distribution; growth of the mean collapse time with K; barriers
from the chain and from the diffusion approximation; Gillespie simulation against
the exact mean collapse time; dependence on theta; immigration."""
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import logsumexp
import model as M

HH, SH = M.game("HG", "HG"), M.game("SD", "HG")


def setup(g, K, theta=0.5, iota=0.0):
    ms, xs = M.attractors(g)[-1]
    birth, death = M.rates(K, M.chain_x(g, K, xs), theta, iota)
    return ms, xs, birth, death


def all_times(birth, death):
    lp = M.log_product(birth, death)
    tail = np.logaddexp.accumulate(lp[::-1])[::-1]
    return np.exp(np.logaddexp.accumulate(tail - lp - np.log(death)))


print("product formula vs exact quasi-stationary distribution")
for g, lab in ((HH, "HG/HG"), (SH, "SD/HG")):
    for K in (20, 60, 100):
        _, _, b, d = setup(g, K)
        u, _ = M.qsd_exact(b, d)
        p = M.qsd_product(b, d)
        T = all_times(b, d)
        Te, Tp = u @ T, p @ T
        print(f"  {lab} K={K}: max state difference {np.abs(u-p).max():.2g}, "
              f"relative difference in mean collapse time {abs(Te-Tp)/Te:.2g}")

print("ln T - K*barrier, harmony in both states")
ms, xs = M.attractors(HH)[-1]
dV = M.barrier(HH, ms, xs, n=400001)[0]
prev = None
for K in (100, 200, 400, 800):
    _, _, b, d = setup(HH, K)
    v = np.log(M.mean_time_to_zero(b, d, int(round(ms * K)))) - K * dV
    print(f"  K={K}: {v:.3f}" + (f"  increment {v-prev:.3f}" if prev is not None else ""))
    prev = v

print("barrier from the chain and from the diffusion approximation")
for g, lab in ((HH, "HG/HG"), (SH, "SD/HG")):
    ms, xs = M.attractors(g)[-1]
    m, x = M.path(g, ms, xs, 400001)
    bp, bm = M.bplus(m, x), M.bminus(m, x)
    chain = trapezoid(np.log(bp / bm), m)
    diff = trapezoid(2 * (bp - bm) / (bp + bm), m)
    print(f"  {lab}: chain {chain:.6f}, diffusion {diff:.6f}")

print("Gillespie simulation vs exact mean collapse time")
rng = np.random.default_rng(1)
zmax = 0
for g, lab in ((HH, "HG/HG"), (SH, "SD/HG")):
    for K, n in ((10, 4000), (20, 3000), (30, 2000), (40, 1200)):
        ms, xs, b, d = setup(g, K)
        M0 = int(round(ms * K))
        T = M.mean_time_to_zero(b, d, M0)
        state = np.full(n, M0)
        t = np.zeros(n)
        alive = np.ones(n, bool)
        while alive.any():
            i = np.nonzero(alive)[0]
            bi, di = b[state[i] - 1], d[state[i] - 1]
            t[i] += rng.exponential(1.0, i.size) / (bi + di)
            up = rng.random(i.size) * (bi + di) < bi
            state[i] = np.minimum(state[i] + np.where(up, 1, -1), len(b))
            alive[i] = state[i] > 0
        z = (t.mean() - T) / (t.std(ddof=1) / np.sqrt(n))
        zmax = max(zmax, abs(z))
        print(f"  {lab} K={K}: exact {T:.4g}, simulated {t.mean():.4g} (n={n}, z={z:+.2f})")
print(f"  largest |z| = {zmax:.2f}")

print("dependence on theta")
for theta in (0.5, 0.75, 1.0):
    out = []
    for g in (HH, SH):
        ms, xs = M.attractors(g)[-1]
        out.append(M.barrier(g, ms, xs, theta=theta, n=400001)[0])
    print(f"  theta={theta}: {out[0]:.5f} {out[1]:.5f}  ratio {out[0]/out[1]:.3f}  difference {out[0]-out[1]:.5f}")

print("immigration: stationary probability of M in {0, 1}")
for g, lab in ((HH, "HG/HG"), (SH, "SD/HG")):
    for K in (20, 60, 100):
        _, xs, b, d = setup(g, K)
        u, _ = M.qsd_exact(b, d)
        crit = 1.0 / (K * (u @ all_times(b, d)))
        row = []
        for iota in (crit / 10, crit, 10 * crit):
            bi = b + iota * K
            lp = np.concatenate(([0.0], np.log(iota * K) + M.log_product(bi, d) - np.log(d[0])))
            p = np.exp(lp - logsumexp(lp))
            row.append(p[:2].sum())
        print(f"  {lab} K={K}: " + "  ".join(f"{r:.3f}" for r in row))
