"""Fig. 5: strategy frequency below equilibrium, V(m) - V(m*), and the exact mean
time to resource collapse, for the four games with harmony in the rich state and
for harmony (poor) / snowdrift (rich)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M

PAIRS = [("HG", "HG", "#08306b", "-"), ("SH", "HG", "#4292c6", (0, (5, 2.5))),
         ("SD", "HG", "#e08214", "-"), ("PD", "HG", "#b2182b", "-"),
         ("HG", "SD", "0.45", (0, (1.5, 1.5)))]
KS = np.array([40, 60, 80, 100, 140, 180, 240, 300])

fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.1))
plt.subplots_adjust(wspace=0.32, left=0.06, right=0.985, bottom=0.15, top=0.92)
for poor, rich, col, ls in PAIRS:
    g = M.game(poor, rich)
    ms, xs = M.attractors(g)[-1]
    dV, m, x, V = M.barrier(g, ms, xs)
    T = np.array([M.mean_time_to_zero(*M.rates(K, M.chain_x(g, K, xs)), int(round(ms * K)))
                  for K in KS])
    lab = f"{M.NAMES[poor]} $\\rightarrow$ {M.NAMES[rich]}"
    axs[0].plot(m, x, color=col, ls=ls, lw=2.0, label=lab)
    axs[1].plot(m, V, color=col, ls=ls, lw=2.0)
    axs[1].plot([m[np.argmax(V)]], [V.max()], "o", color=col, ms=5, mec="w", mew=0.7, zorder=5)
    axs[1].text(ms + 0.012, dV, f"{dV:.3f}", color=col, fontsize=8, va="center")
    A = np.exp(np.mean(np.log(T) - KS * dV))
    Kf = np.linspace(KS[0], KS[-1], 200)
    axs[2].plot(KS, T, "o", color=col, ms=4.2, mfc=col if ls == "-" else "none", mew=1.1)
    axs[2].plot(Kf, A * np.exp(Kf * dV), color=col, ls=ls, lw=1.1)
    print(f"{poor}/{rich}: m*={ms:.4f} x*={xs:.4f} barrier={dV:.5f} T(K=100)={T[3]:.4g}")
axs[0].plot([0.84], [1.0], "s", ms=9, color="k", zorder=6)
axs[0].plot([0.7067], [0.8189], "s", ms=8, mfc="w", mec="0.45", mew=1.4, zorder=6)
axs[0].set_xlabel(r"resource level, $m$")
axs[0].set_ylabel(r"cooperator frequency $x_m$ below equilibrium")
axs[0].set_xlim(0, 0.9)
axs[0].set_ylim(-0.05, 1.1)
axs[0].legend(loc="lower right", fontsize=7.2, frameon=False, handlelength=2.4)
axs[1].set_xlabel(r"resource level, $m$")
axs[1].set_ylabel(r"$V(m)-V(m^*)$")
axs[1].set_xlim(0, 0.95)
axs[1].axhline(0, color="0.8", lw=0.6)
axs[2].set_yscale("log")
axs[2].set_xlabel(r"carrying capacity, $K$")
axs[2].set_ylabel(r"mean time to collapse, $T_{\rm coll}$ (units of $1/s\epsilon$)")
for ax, l in zip(axs, "abc"):
    ax.text(-0.14, 1.03, l, transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.savefig("figures/fig5_collapse.png", dpi=220)
