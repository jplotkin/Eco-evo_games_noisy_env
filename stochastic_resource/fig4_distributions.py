"""Fig. 4: distributions of the resource level m and the cooperator frequency x for
harmony (poor) / snowdrift (rich), K = 20, 150, 1000. (a) eps -> 0, analytical
distributions (SI Eqs. 8-9); (b) eps = 0.1 and (c) eps = 10, histograms from long
simulations of the coupled system. Run with --as-submitted for the submitted labels."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sde

AS_SUBMITTED = "--as-submitted" in sys.argv
G = (3.0, 1.0, 2.0, -3.0)
KS = [20, 150, 1000]
CM = ["darkolivegreen", "tab:green", "palegreen"]
CX = ["darkblue", "tab:blue", "lightskyblue"]
BIN = 0.02
M_THR = G[0] / (G[0] + G[2])                     # full cooperation stable below m = 0.6
T = 50000.0


def x_rest(m):
    return sde.rest_point(1.0, m, *G)


def analytical(K):
    mg = np.linspace(1e-4, 10, 400001)
    xm = np.array([x_rest(mm) for mm in mg])
    f = sde.stationary_m(G, K, mg, lambda mm: np.interp(mm, mg, xm))
    w = f * np.gradient(mg)
    w /= w.sum()
    xedges = np.arange(0, 1 + BIN, BIN)
    px = np.histogram(np.clip(xm, 0, 1 - 1e-12), bins=xedges, weights=w)[0] / BIN
    return mg, f, xedges, px


fig, axs = plt.subplots(2, 3, figsize=(18, 9.5))
plt.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.08, wspace=0.18, hspace=0.3)
titles = [r"$\epsilon\rightarrow 0$", r"$\epsilon = 0.1$", r"$\epsilon = 10$"]
for c, eps in enumerate([None, 0.1, 10.0]):
    axm, axx = axs[0, c], axs[1, c]
    for k, K in enumerate(KS):
        lab = f"K={K}" if AS_SUBMITTED else f"$K = {K}$"
        if eps is None:
            mg, f, xedges, px = analytical(K)
            axm.plot(mg, f, color=CM[k], lw=3, label=lab)
            cx = 0.5 * (xedges[1:] + xedges[:-1])
            nz = np.nonzero(px)[0]
            sl = slice(nz.min(), nz.max() + 1)
            axx.plot(cx[sl], px[sl], color=CX[k], lw=3, label=lab)
        else:
            dt = min(0.01, 0.01 / eps)
            run = sde.coupled(*G, float(K), eps, 1.0, T, dt, 0.9, 0.9, max(1, int(0.1 / dt)), 31 + k + int(10 * eps))
            xs, ms = run[:, 1], run[:, 2]
            medges = np.arange(0, 1.5 + BIN, BIN)
            xedges = np.arange(0, 1 + BIN, BIN)
            for ax, data, edges, col in ((axm, ms, medges, CM[k]), (axx, xs, xedges, CX[k])):
                ax.hist(np.clip(data, edges[0], edges[-1] - 1e-9), bins=edges, density=True,
                        color=col, alpha=0.2, lw=0)
                ax.hist(np.clip(data, edges[0], edges[-1] - 1e-9), bins=edges, density=True,
                        histtype="step", color=col, lw=2, label=lab)
            print(f"eps={eps} K={K}: fraction of time with m < 0.6: {np.mean(ms < M_THR):.3f}")
    axm.axvline(M_THR, color="tab:blue", ls="--", lw=1.2)
    axm.set_xlim(-0.02, 1.5)
    axm.set_ylim(0, 12)
    axx.set_xlim(-0.02, 1.02)
    axx.set_ylim(0, 50)
    axm.set_title(titles[c], fontsize=20, fontweight="bold")
    axm.text(-0.08, 1.06, "abc"[c], transform=axm.transAxes, fontsize=20, fontweight="bold")
    for ax in (axm, axx):
        ax.legend(loc="upper left", fontsize=11)
    if AS_SUBMITTED:
        axm.set_xlabel("m")
        axx.set_xlabel("x")
    else:
        axm.set_xlabel(r"resource level, $m$", fontsize=13)
        axx.set_xlabel(r"cooperator frequency, $x$", fontsize=13)
        axm.set_ylabel("probability density", fontsize=13)
        axx.set_ylabel("probability density", fontsize=13)
fig.savefig("figures/fig4_distributions" + ("_as_submitted" if AS_SUBMITTED else "") + ".png", dpi=170)
