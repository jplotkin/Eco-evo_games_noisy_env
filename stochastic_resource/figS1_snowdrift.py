"""SI Fig. S1: harmony (poor) / snowdrift (rich) with dH0 = 1, DH1 = -3 and the pair
(dL0, DL1) varied, so that the fraction of the effective game that is a snowdrift game,
DL1 / (dL0 + DL1), runs from 0.2 to 0.8. (a, b) Phase plane and distributions of m and
x for two pairs, K = 1000, timescale separation. (c-k) Fraction of time the cooperator
frequency is interior (0.01 < x < 0.99) against the snowdrift fraction, for K = 1000,
100, 10 (rows) and eps -> 0, 0.1, 1 (columns). Run with --as-submitted for the
submitted axis labels."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sde
import panels

AS_SUBMITTED = "--as-submitted" in sys.argv
PAIRS = [(4, 1), (3, 1), (2, 1), (3, 2), (4, 3), (1, 1), (3, 4), (2, 3), (1, 2), (1, 3), (1, 4)]
KS = [1000, 100, 10]
EPS = [None, 0.1, 1.0]
T, DT = 20000.0, 0.01
BIN = 0.02


def game(dL0, DL1):
    return (float(dL0), 1.0, float(DL1), -3.0)


def interior_fraction(g, K, eps, seed):
    m0 = 0.9
    if eps is None:
        run = sde.separated(*g, float(K), 1.0, T, DT, m0, 1.0, 10, seed)
    else:
        run = sde.coupled(*g, float(K), eps, 1.0, T, DT, 0.9, m0, 10, seed)
    x = run[len(run) // 10:, 1]
    return np.mean((x > 0.01) & (x < 0.99))


fig = plt.figure(figsize=(16, 17))
top = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.96, bottom=0.71, wspace=0.22)
for c, (dL0, DL1) in enumerate([(4, 1), (1, 4)]):
    g = game(dL0, DL1)
    cell = top[c].subgridspec(2, 2, width_ratios=[1.25, 1], wspace=0.35, hspace=0.6)
    ax = fig.add_subplot(cell[:, 0])
    panels.phase_plane(ax, g, AS_SUBMITTED)
    fig.text(top[c].get_position(fig).x0 - 0.06, 0.975, "ab"[c], fontsize=22, fontweight="bold", va="top")
    run = sde.separated(*g, 1000.0, 1.0, 200000.0, 0.01, 0.9, 1.0, 100, 50 + c)
    xs, ms = run[:, 1], run[:, 2]
    axm, axx = fig.add_subplot(cell[0, 1]), fig.add_subplot(cell[1, 1])
    axm.hist(ms, bins=np.arange(0, 1.2 + BIN / 2, BIN / 2), density=True, color="tab:green")
    axm.axvline(dL0 / (dL0 + DL1), color="k", ls="--", lw=1)
    mg = np.linspace(1e-4, 10, 200001)
    xm = np.array([sde.rest_point(1.0, mm, *g) for mm in mg])
    f = sde.stationary_m(g, 1000, mg, lambda mm: np.interp(mm, mg, xm))
    axm.plot(mg, f, color="k", lw=1.6)
    w = f * np.gradient(mg)
    w /= w.sum()
    xe = np.arange(0, 1 + BIN, BIN)
    px = np.histogram(np.clip(xm, 0, 1 - 1e-12), bins=xe, weights=w)[0] / BIN
    cx = 0.5 * (xe[1:] + xe[:-1])
    nz = np.nonzero(px)[0]
    inner = nz[(nz > 0) & (nz < len(px) - 1)]
    axx.plot(cx[inner], px[inner], color="k", lw=1.6)
    for end in (0, len(px) - 1):
        if px[end] > 0.05:
            axx.plot([cx[end]], [px[end]], "s", color="k", ms=5)
    axx.hist(np.clip(xs, 0, 1 - 1e-12), bins=np.arange(0, 1 + BIN, BIN), density=True, color="tab:blue")
    axm.set_xlim(0, 1.2)
    axx.set_xlim(0, 1.02)
    axm.set_xlabel(r"resource level, $m$" if not AS_SUBMITTED else "m")
    axx.set_xlabel(r"cooperator frequency, $x$" if not AS_SUBMITTED else "x")
    if not AS_SUBMITTED:
        axm.set_ylabel("probability density")
        axx.set_ylabel("probability density")
    print(f"panel {'ab'[c]}: snowdrift fraction {DL1/(dL0+DL1):.2f}, fraction of time at x=1 {np.mean(xs == 1.0):.3f}")

grid = fig.add_gridspec(3, 3, left=0.12, right=0.98, top=0.64, bottom=0.05, wspace=0.12, hspace=0.18)
frac = np.array([DL1 / (dL0 + DL1) for dL0, DL1 in PAIRS])
for i, K in enumerate(KS):
    for j, eps in enumerate(EPS):
        ax = fig.add_subplot(grid[i, j])
        y = np.array([interior_fraction(game(*p), K, eps, 1000 * i + 100 * j + n) for n, p in enumerate(PAIRS)])
        ax.plot(frac, y, "x", color="k", ms=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.05)
        ax.text(0.02, 0.97, "cdefghijk"[3 * i + j], transform=ax.transAxes, fontsize=18,
                fontweight="bold", va="top")
        if i == 0 and j == 0:
            ax.plot([0.2], [y[0]], "s", ms=18, mfc="none", mec="darkorange", mew=2)
            ax.plot([0.8], [y[-1]], "s", ms=18, mfc="none", mec="tab:purple", mew=2)
        if i == 0 and j == 2:
            ax.text(0.6, 0.45, "Tilman et al., 2020", fontsize=15, style="italic", ha="center")
        if i == 0:
            ax.set_title([r"$\epsilon\rightarrow 0$", r"$\epsilon = 0.1$", r"$\epsilon = 1$"][j], fontsize=17)
        if j == 0:
            ax.set_ylabel(f"$K = {K}$\n\nfraction of time\nwith interior $x$" if not AS_SUBMITTED
                          else "Proportion of time in interior state", fontsize=12)
        else:
            ax.set_yticklabels([])
        if i == 2:
            ax.set_xlabel("fraction of the effective game\nthat is a snowdrift game" if not AS_SUBMITTED
                          else "Proportion of effective game that is snowdrift", fontsize=12)
        print(f"K={K} eps={eps}: " + " ".join(f"{v:.2f}" for v in y))
fig.savefig("figures/figS1_snowdrift" + ("_as_submitted" if AS_SUBMITTED else "") + ".png", dpi=160)
