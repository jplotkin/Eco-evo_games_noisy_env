"""Fig. 6: (a) fraction of time at full cooperation against K for harmony (poor) /
snowdrift (rich), with the Gaussian rule and K_noise; (b) K Var(m) at K = 8000
against the feedback strength phi, over a grid of incentive parameters."""
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import model as M

FB = M.Q * (M.EH - M.EL) / M.R
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.3))
plt.subplots_adjust(wspace=0.28, left=0.075, right=0.985, bottom=0.14, top=0.93)

# (a)
g = M.game("HG", "SD")
ms, xs = M.attractors(g)[-1]
m_thr = g[0] / (g[0] + g[2])                      # full cooperation stable below m_L
kvar = M.variance_formula(ms, M.interior_slope(ms, g))
Ks = np.unique(np.round(np.logspace(np.log10(30), np.log10(3000), 26)).astype(int))
frac = []
for K in Ks:
    xk = M.chain_x(g, K, xs)
    u, _ = M.qsd_exact(*M.rates(K, xk))
    m = np.arange(1, len(u) + 1) / K
    frac.append(u[m < m_thr].sum())
frac = np.array(frac)
gauss = norm.cdf((m_thr - ms) / np.sqrt(kvar / Ks))
K_noise = 4 * kvar / (ms - m_thr) ** 2
ax = axs[0]
ax.loglog(Ks, frac, "o", ms=5, color="#08306b", label="exact (quasi-stationary)")
ax.loglog(Ks, gauss, "-", color="#b2182b", lw=1.3, label=r"$\Phi[(m_{\rm thr}-m^*)/\sigma_m]$")
ax.axvline(K_noise, color="0.45", ls=":", lw=1.1)
ax.text(K_noise * 1.12, 2e-4, r"$K_{\rm noise}=$" + f"{K_noise:.0f}", fontsize=9, color="0.25")
ax.set_xlabel(r"environmental carrying capacity, $K$")
ax.set_ylabel("fraction of time spent at full cooperation")
ax.set_ylim(1e-5, 1.4)
ax.legend(loc="lower left", frameon=False)
print(f"(a) m*={ms:.4f} m_thr={m_thr} K*Var={kvar:.4f} K_noise(z=2)={K_noise:.1f}")
for K in (50, 150, 1000):
    u, _ = M.qsd_exact(*M.rates(K, M.chain_x(g, K, xs)))
    m = np.arange(1, len(u) + 1) / K
    print(f"    K={K}: fraction at full cooperation {u[m < m_thr].sum():.4g}")

# (b)
K = 8000
grid = itertools.product([-0.5, -0.75, -1, -1.5, -2, -2.5, -3, -4, -5],
                         [0.5, 1, 1.5, 2, 3, 4], [0.5, 1, 1.5, 2, 3, 4],
                         [-0.5, -1, -1.5, -2, -3, -4])
pts, n_grid = [], 0
for gg in grid:
    n_grid += 1
    att = M.attractors(gg)
    if not att or not 0 < att[-1][1] < 1:
        continue
    ms_, xs_ = att[-1]
    slope = M.interior_slope(ms_, gg)
    phi = FB * slope
    p = M.qsd_product(*M.rates(K, M.chain_x(gg, K, xs_)))
    m = np.arange(1, len(p) + 1) / K
    var = float(np.sum(p * m ** 2) - np.sum(p * m) ** 2)
    pts.append((phi, K * var, M.variance_formula(ms_, slope)))
pts = np.array(pts)
ax = axs[1]
ph = np.linspace(-1.6, 0.94, 400)
for m0, ls in ((0.3, "--"), (0.9, "-")):
    ax.plot(ph, ((M.LAM + M.MU) + M.R * m0) / (2 * M.R * (1 - ph)), ls, color="0.35", lw=1.1,
            label=f"closed form, $m^*={m0}$")
ax.plot(pts[:, 0], pts[:, 1], "o", ms=4.2, color="#08306b", label=f"exact, $K={K}$")
ax.axvline(1.0, color="#b2182b", ls=":", lw=1.2)
ax.text(0.96, 26, "deterministic\nstability boundary", fontsize=8, color="#b2182b", ha="right", va="top")
ax.set_xlabel(r"eco-evolutionary feedback, $\varphi=q(e_H-e_L)x'(m^*)/(\lambda-\mu)$")
ax.set_ylabel(r"scaled environmental variance, $K\,{\rm Var}(m)$")
ax.set_xlim(-1.7, 1.12)
ax.set_ylim(0, 30)
ax.legend(loc="upper left", frameon=False)
for a_, l in zip(axs, "ab"):
    a_.text(-0.14, 1.03, l, transform=a_.transAxes, fontsize=13, fontweight="bold")
err = np.abs(pts[:, 1] / pts[:, 2] - 1)
print(f"(b) {n_grid} grid points, {len(pts)} with an interior attractor; "
      f"max relative discrepancy {err.max():.4f}; phi in [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]")
fig.savefig("figures/fig6_noise.png", dpi=220)
