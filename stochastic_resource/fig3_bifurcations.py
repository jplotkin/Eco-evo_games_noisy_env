"""Fig. 3: bifurcation diagrams of the strategy rest points against the resource level
for all combinations of the game in a poor environment (rows) and in a rich environment
(columns). Magnitudes |dL0| = 3, |dH0| = 1, |DL1| = 2, |DH1| = 3; four cells add a
second diagram with |dL0| = 1, |dH0| = 3, |DL1| = 3, |DH1| = 2."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M
import panels

COLORS = {"PD": "#ffcccc", "SD": "#e3d9ec", "SH": "#e5f8cf", "HG": "#d3e2ee"}
ROW_TXT = {"PD": (r"$\delta_L^0<0$ and $\delta_H^0<0$", "Prisoner's dilemma"),
           "SD": (r"$\delta_L^0<0$ and $\delta_H^0>0$", "Snowdrift game"),
           "SH": (r"$\delta_L^0>0$ and $\delta_H^0<0$", "Stag-hunt game"),
           "HG": (r"$\delta_L^0>0$ and $\delta_H^0>0$", "Harmony game")}
COL_TXT = {"PD": (r"$\Delta_L^1>0$ and $\Delta_H^1>0$", "Prisoner's dilemma"),
           "SD": (r"$\Delta_L^1>0$ and $\Delta_H^1<0$", "Snowdrift game"),
           "SH": (r"$\Delta_L^1<0$ and $\Delta_H^1>0$", "Stag-hunt game"),
           "HG": (r"$\Delta_L^1<0$ and $\Delta_H^1<0$", "Harmony game")}
TWO = {("PD", "HG"), ("SD", "SH"), ("SH", "SD"), ("HG", "PD")}
ALT = (1, 3, 3, 2)

fig = plt.figure(figsize=(17, 12))
gs = fig.add_gridspec(5, 5, width_ratios=[0.55, 1, 1, 1, 1], height_ratios=[0.32, 1, 1, 1, 1],
                      wspace=0.0, hspace=0.0, left=0.03, right=0.995, top=0.94, bottom=0.01)
fig.text(0.5 + 0.55 / 4.55 / 2, 0.965, "Game 1 (resource-rich environment)", ha="center", fontsize=15)
fig.text(0.012, 0.47, "Game 0 (resource-poor environment)", rotation=90, va="center", fontsize=15)
for j, rich in enumerate(M.GAMES):
    ax = fig.add_subplot(gs[0, j + 1])
    ax.set_facecolor(COLORS[rich])
    ax.set_xticks([]), ax.set_yticks([])
    ax.text(0.5, 0.62, COL_TXT[rich][0], ha="center", va="center", fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.25, COL_TXT[rich][1], ha="center", va="center", fontsize=14, transform=ax.transAxes)
for i, poor in enumerate(M.GAMES):
    ax = fig.add_subplot(gs[i + 1, 0])
    ax.set_facecolor(COLORS[poor])
    ax.set_xticks([]), ax.set_yticks([])
    ax.text(0.5, 0.62, ROW_TXT[poor][0].replace(" and ", "\nand\n"), ha="center", va="center",
            fontsize=13, transform=ax.transAxes)
    ax.text(0.5, 0.18, ROW_TXT[poor][1].replace(" ", "\n", 1), ha="center", va="center",
            fontsize=13, transform=ax.transAxes)
    for j, rich in enumerate(M.GAMES):
        frame = fig.add_subplot(gs[i + 1, j + 1])
        frame.set_xticks([]), frame.set_yticks([])
        games = [M.game(poor, rich)]
        if (poor, rich) in TWO:
            games.append(M.game(poor, rich, ALT))
        inner = gs[i + 1, j + 1].subgridspec(len(games), 1, hspace=0.55 if len(games) == 2 else 0.0)
        for k, g in enumerate(games):
            sub = inner[k].subgridspec(1, 1)
            ax = fig.add_subplot(sub[0])
            pos = ax.get_position()
            shrink = 0.16 if len(games) == 1 else 0.25
            ax.set_position([pos.x0 + 0.17 * pos.width, pos.y0 + shrink * pos.height,
                             0.78 * pos.width, (0.92 - shrink) * pos.height])
            panels.bifurcation(ax, g, compact=True, marker_size=7 if len(games) == 2 else 9,
                               show_thresholds=False)
            ax.tick_params(labelsize=8)
            if len(games) == 2:
                ax.set_yticks([0, 0.5, 1])
fig.savefig("figures/fig3_bifurcations.png", dpi=170)
