"""Panels shared by Figs. 2, 3 and S1: the dilemma phase plane and the bifurcation
diagram of the strategy rest points against the resource level."""
import numpy as np
import sde

M_EQ0 = 1 - sde.Q * sde.EH / sde.R            # resource level with x = 0
M_EQ1 = 1 - sde.Q * sde.EL / sde.R            # resource level with x = 1


def thresholds(g):
    dL0, dH0, DL1, DH1 = g
    out = []
    for num, den in ((dL0, dL0 + DL1), (dH0, dH0 + DH1)):
        if den != 0 and 0 < num / den < 1:
            out.append(num / den)
    return out


def phase_plane(ax, g, AS_SUBMITTED=False):
    dL0, dH0, DL1, DH1 = g
    lim = 4.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(range(-4, 5))
    ax.set_yticks(range(-4, 5))
    ax.grid(alpha=0.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.plot([DH1, -dH0], [DL1, -dL0], "-x", color="g", lw=2, ms=9)
    taken = []
    for name, (px, py), qy in (("game 1", (DH1, DL1), -dL0), ("game 0", (-dH0, -dL0), DL1)):
        # label on the side of the point away from the line, inside the axes
        dy = -13 if py > 3.5 else (5 if py < -3.5 else (5 if qy < py else -13))
        taken.append(ax.annotate(name, (px, py), color="g", xytext=(5, dy), textcoords="offset points",
                                 fontsize=10))
    # quadrant names, each at the first position that clears the line between the games
    ax.apply_aspect()
    renderer = ax.figure.canvas.get_renderer()
    t = np.linspace(0, 1, 400)
    line = ax.transData.transform(np.column_stack([DH1 + t * (-dH0 - DH1), DL1 + t * (-dL0 - DL1)]))
    boxes = [a.get_window_extent(renderer).padded(3) for a in taken]
    for (x0, x1, y0, y1, c, lab, sx, sy) in [(-lim, 0, 0, lim, "rebeccapurple", "snowdrift", -1, 1),
                                             (0, lim, 0, lim, "red", "prisoner's dilemma", 1, 1),
                                             (-lim, 0, -lim, 0, "tab:blue", "harmony", -1, -1),
                                             (0, lim, -lim, 0, "lawngreen", "stag hunt", 1, -1)]:
        ax.fill_between([x0, x1], y0, y1, color=c, alpha=0.2, lw=0)
        if AS_SUBMITTED:
            lab = {"snowdrift": "SD", "harmony": "HARMONY", "stag hunt": "SH"}.get(lab, "PD")
        for tx, ty in ((2.25, 4.0), (2.25, 0.45), (2.25, 2.2), (3.3, 4.0), (3.3, 0.45), (1.2, 4.0), (1.2, 0.45)):
            txt = ax.text(sx * tx, sy * ty, lab, ha="center", va="center", fontsize=11 if AS_SUBMITTED else 8.5)
            bb = txt.get_window_extent(renderer).padded(2)
            hit = bb.count_contains(line) > 0 or any(bb.overlaps(b) for b in boxes)
            if not hit:
                break
            txt.remove()
        else:
            ax.text(sx * 2.25, sy * 4.0, lab, ha="center", va="center", fontsize=11 if AS_SUBMITTED else 8.5)
    if AS_SUBMITTED:
        ax.set_xlabel("Delta_1H, -delta_0H")
        ax.set_ylabel("Delta_1L, -delta_0L")
    else:
        ax.set_xlabel(r"$\Delta_H^1,\ -\delta_H^0$")
        ax.set_ylabel(r"$\Delta_L^1,\ -\delta_L^0$")


def bifurcation(ax, g, AS_SUBMITTED=False, compact=False, marker_size=10, show_thresholds=True):
    m = np.linspace(0, 1, 400)
    a = np.array([sde.D(1.0, mm, *map(float, g)) for mm in m])
    b = np.array([sde.D(0.0, mm, *map(float, g)) for mm in m])
    for xs, stab, col in ((np.zeros_like(m), b < 0, "tab:red"), (np.ones_like(m), a > 0, "tab:blue")):
        ax.plot(np.where(stab, m, np.nan), np.where(stab, xs, np.nan), "-", color=col, lw=3.5)
        ax.plot(np.where(~stab, m, np.nan), np.where(~stab, xs, np.nan), ":", color=col, lw=1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        x2 = b / (b - a)
    ok = (x2 > 0) & (x2 < 1)
    st = ok & (b > 0) & (a < 0)
    ax.plot(np.where(st, m, np.nan), np.where(st, x2, np.nan), "-", color="tab:purple", lw=3.5)
    ax.plot(np.where(ok & ~st, m, np.nan), np.where(ok & ~st, x2, np.nan), ":", color="tab:purple", lw=1.5)
    # equilibria of the coupled deterministic system
    gf = tuple(map(float, g))
    ax.plot([M_EQ0], [0], "s", ms=marker_size, color="tab:red",
            mfc="tab:red" if sde.D(0.0, M_EQ0, *gf) < 0 else "white")
    ax.plot([M_EQ1], [1], "s", ms=marker_size, color="tab:blue",
            mfc="tab:blue" if sde.D(1.0, M_EQ1, *gf) > 0 else "white")
    x2 = np.where(ok, x2, 0.0)
    f = np.where(ok, sde.R * (1 - m) - sde.Q * (sde.EL * x2 + sde.EH * (1 - x2)), np.nan)
    for i in range(len(m) - 1):
        if ok[i] and ok[i + 1] and np.sign(f[i]) != np.sign(f[i + 1]):
            mm = m[i] - f[i] * (m[i + 1] - m[i]) / (f[i + 1] - f[i])
            filled = st[i] and f[i] > 0
            ax.plot([mm], [x2[i]], "s", ms=marker_size + 1, color="tab:purple", mfc="tab:purple" if filled else "white")
    if show_thresholds:
        for th in thresholds(g):
            ax.axvline(th, color="k", ls="--", lw=0.8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.1, 1.1)
    if compact:
        ax.set_xlabel(r"$m$", labelpad=1)
        ax.set_ylabel(r"$x^*$", labelpad=1)
    else:
        ax.set_xlabel(r"resource level, $m$" if not AS_SUBMITTED else "m")
        ax.set_ylabel(r"equilibrium cooperator" "\n" r"frequency, $x^*$" if not AS_SUBMITTED else "x*")


