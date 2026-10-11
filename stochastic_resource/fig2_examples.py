"""Fig. 2: four combinations of games under timescale separation, K = 100. Columns:
dilemma phase plane; bifurcation diagram of the strategy rest points against m;
trajectories of x and m; long-run distributions of m and x (simulation histograms and
the analytical distributions of SI Eqs. 8-9). Distributions are shown as the fraction
of time per bin of width 0.02. Run with --as-submitted for the submitted layout
(densities, string axis labels)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sde
from panels import thresholds, M_EQ0, M_EQ1
import panels

AS_SUBMITTED = "--as-submitted" in sys.argv
# (dL0, dH0, DL1, DH1), initial resource level, initial strategy frequency
ROWS = [((3, 1, 2, -3), 0.4, 1.0, "harmony / snowdrift"),
        ((3, 1, 2, 3), 0.8, 0.0, "harmony / prisoner's dilemma"),
        ((1, 3, 3, 2), 0.45, 0.5, "harmony / prisoner's dilemma"),
        ((-3, -1, -2, 3), 0.9, 1.0, "prisoner's dilemma / stag hunt")]
K = 100
BIN = 0.02


def path_x(g, m, x0):
    """Strategy frequency along a sequence of resource levels, starting from x0."""
    out, x = np.empty(len(m)), x0
    for i, mm in enumerate(m):
        x = sde.rest_point(x, mm, *map(float, g))
        out[i] = x
    return out


def analytical(g, x_long_run):
    """Per-bin probabilities of m and x from the diffusion approximation (SI Eq. 8),
    with x = x_m along the rest points the population occupies in the long run."""
    mg = np.linspace(1e-4, 10, 200001)
    if x_long_run is None:
        xm = path_x(g, mg[::-1], 1.0)[::-1]
    else:
        xm = np.full_like(mg, x_long_run)
    f = sde.stationary_m(g, K, mg, lambda mm: np.interp(mm, mg, xm))
    w = f * np.gradient(mg)
    w /= w.sum()
    edges = np.arange(0, 1.2 + BIN, BIN)
    pm = np.histogram(mg, bins=edges, weights=w)[0]
    xedges = np.arange(0, 1 + BIN, BIN)
    px = np.histogram(np.clip(xm, 0, 1 - 1e-12), bins=xedges, weights=w)[0]
    return edges, pm, xedges, px


ncol_w = [1.0, 1.05, 1.15, 1.25]
fig = plt.figure(figsize=(18, 17))
outer = fig.add_gridspec(4, 4, width_ratios=ncol_w, wspace=0.42, hspace=0.38,
                         left=0.075, right=0.985, top=0.95, bottom=0.04)
titles = ["Dilemma phase plane", "Bifurcation diagram", "Trajectories over time",
          "Quasi-stationary distributions"]
for r, (g, m0, x0, name) in enumerate(ROWS):
    gf = tuple(map(float, g))
    ax = fig.add_subplot(outer[r, 0])
    panels.phase_plane(ax, g, AS_SUBMITTED)
    ax.text(-0.38, 1.0, "abcd"[r], transform=ax.transAxes, fontsize=24, fontweight="bold", va="top")
    ax2 = fig.add_subplot(outer[r, 1])
    panels.bifurcation(ax2, g, AS_SUBMITTED)
    # trajectory
    ax3 = fig.add_subplot(outer[r, 2])
    tr = sde.separated(*gf, float(K), 0.1, 500.0, 0.01, m0, x0, 10, 7 + r)
    ax3.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=1.2, label=r"cooperator frequency, $x$")
    ax3.plot(tr[:, 0], tr[:, 2], color="tab:green", lw=1.0, label=r"resource level, $m$")
    for th in thresholds(g):
        ax3.axhline(th, color="k", ls="--", lw=0.8)
    ax3.set_ylim(-0.02, 1.25)
    ax3.set_xlabel("time")
    ax3.set_ylabel(r"$x$ and $m$")
    if not AS_SUBMITTED and r == 0:
        ax3.legend(loc="upper right", fontsize=9, framealpha=0.9)
    # long-run distributions
    run = sde.separated(*gf, float(K), 1.0, 200000.0, 0.01, m0, x0, 100, 100 + r)
    xs, ms = run[:, 1], run[:, 2]
    inner = outer[r, 3].subgridspec(2, 1, hspace=0.55)
    axm, axx = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
    medges = np.arange(0, 1.2 + BIN, BIN)
    xedges = np.arange(0, 1 + BIN, BIN)
    hm = np.histogram(np.clip(ms, 0, 1.2 - 1e-9), bins=medges)[0] / len(ms)
    hx = np.histogram(np.clip(xs, 0, 1 - 1e-12), bins=xedges)[0] / len(xs)
    if AS_SUBMITTED:
        hm, hx = hm / BIN, hx / BIN
    axm.bar(medges[:-1], hm, width=BIN, align="edge", color="tab:green")
    axx.bar(xedges[:-1], hx, width=BIN, align="edge", color="tab:blue")
    px = np.zeros_like(hx)
    if r != 1:                                   # hysteresis: no single-valued x_m in row b
        _, pm, _, px = analytical(g, 0.0 if r == 3 else None)
        if AS_SUBMITTED:
            pm, px = pm / BIN, px / BIN
        cm = 0.5 * (medges[1:] + medges[:-1])
        axm.plot(cm, pm, color="k", lw=1.8)
        cx = 0.5 * (xedges[1:] + xedges[:-1])
        inner_bins = (px > 0)
        inner_bins[0] = inner_bins[-1] = False
        axx.plot(cx[inner_bins], px[inner_bins], color="k", lw=1.8)
        for end in (0, -1):
            if px[end] > 1e-3:
                axx.plot([cx[end]], [px[end]], "s", color="k", ms=5)
    for th in thresholds(g):
        axm.axvline(th, color="k", ls="--", lw=0.8)
    axm.set_xlim(0, 1.2)
    axx.set_xlim(-0.01, 1.01)
    axm.set_xlabel(r"resource level, $m$" if not AS_SUBMITTED else "m")
    axx.set_xlabel(r"cooperator frequency, $x$" if not AS_SUBMITTED else "x")
    if not AS_SUBMITTED:
        axm.set_ylabel("fraction of time")
        axx.set_ylabel("fraction of time")
        top = max(hx.max(), px.max() if r != 1 else 0)
        axx.set_ylim(0, 1.15 * top)
    print(f"row {'abcd'[r]}: time-averaged m {ms.mean():.3f}, x {xs.mean():.3f}, "
          f"fraction of time at x=1 {np.mean(xs == 1.0):.3f}, at x=0 {np.mean(xs == 0.0):.3f}")
for c, t in enumerate(titles):
    x = sum(ncol_w[:c]) / sum(ncol_w)
    fig.text(0.06 + 0.925 * (x + 0.5 * ncol_w[c] / sum(ncol_w)), 0.975, t, ha="center",
             fontsize=17, fontweight="bold")
fig.savefig("figures/fig2_examples" + ("_as_submitted" if AS_SUBMITTED else "") + ".png", dpi=170)
