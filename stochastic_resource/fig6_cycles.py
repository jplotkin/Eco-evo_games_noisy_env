"""Fig. 6: the oscillating tragedy of the commons (harmony poor / prisoner's dilemma rich)
in a deterministic and in a noisy environment. Rows: K = infinity, 10000, 100.
Columns: eps = 0.5, 1.8, 2. Each panel shows one run and reports the population mean
payoff averaged over time and, in the noisy rows, over NRUN independent runs.
Run with --as-submitted for the layout of the submitted version (single-run payoffs)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sde

AS_SUBMITTED = "--as-submitted" in sys.argv
G = (3.0, 1.0, 2.0, 3.0)
KS = [0, 10000, 100]                     # 0 = deterministic
EPS = [0.5, 1.8, 2.0]
IC = [(0.2, 0.3), (0.5, 0.7), (0.5, 0.9)]
T, DT = 500.0, 0.01
NRUN = 40

fig, axs = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey=True)
plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.07, wspace=0.08, hspace=0.12)
for i, K in enumerate(KS):
    for j, eps in enumerate(EPS):
        if K == 0:
            def rhs(t, z):
                x_ = 1 / (1 + np.exp(-z[0]))
                return [sde.D(x_, z[1], *G),
                        eps * z[1] * (sde.R * (1 - z[1]) - sde.Q * (sde.EL * x_ + sde.EH * (1 - x_)))]
            t = np.linspace(0, T, 5001)
            z0 = [np.log(IC[j][0] / (1 - IC[j][0])), IC[j][1]]
            y_, m = solve_ivp(rhs, (0, T), z0, t_eval=t, rtol=1e-9, atol=1e-11).y
            x = 1 / (1 + np.exp(-y_))
            w = np.array([sde.mean_payoff(a, b, *G) for a, b in zip(x, m)])
        else:
            t, x, m, w = sde.coupled(*G, float(K), eps, 1.0, T, DT, IC[j][0], IC[j][1], 10, 100 * i + j).T
        ax = axs[i, j]
        ax.plot(t, x, color="tab:blue", lw=1.6, label="cooperator frequency, $x$")
        ax.plot(t, m, color="tab:green", lw=1.0 if K else 1.6, label="resource level, $m$")
        ax.set_ylim(0, 1.25)
        if AS_SUBMITTED or K == 0:
            wbar = w.mean()
        else:
            wbar = np.mean([sde.coupled(*G, float(K), eps, 1.0, T, DT, IC[j][0], IC[j][1], 10,
                                        10000 * (i + 1) + 100 * j + n)[:, 3].mean() for n in range(NRUN)])
        label = f"{wbar:.3f}" if AS_SUBMITTED else f"mean payoff {wbar:.2f}"
        ax.text(0.97, 0.95, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=12 if not AS_SUBMITTED else 15, fontweight="bold" if AS_SUBMITTED else None)
        ax.text(0.01, 0.95, "abcdefghi"[3 * i + j], transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top")
        print(f"K={K or 'inf'} eps={eps}: mean payoff {wbar:.3f}")
for j, eps in enumerate(EPS):
    axs[0, j].set_title(rf"$\epsilon = {eps}$", fontsize=15)
    axs[2, j].set_xlabel("time", fontsize=13)
for i, K in enumerate(KS):
    lab = r"$K\rightarrow\infty$" if K == 0 else f"$K = {K:,}$".replace(",", "{,}")
    if AS_SUBMITTED:
        axs[i, 0].set_ylabel("x and m", fontsize=13)
    else:
        axs[i, 0].set_ylabel(lab + "\n\n$x$ and $m$", fontsize=13)
if not AS_SUBMITTED:
    axs[0, 2].legend(loc="lower right", fontsize=10, frameon=True)
fig.savefig("figures/fig6_cycles" + ("_as_submitted" if AS_SUBMITTED else "") + ".png", dpi=200)
