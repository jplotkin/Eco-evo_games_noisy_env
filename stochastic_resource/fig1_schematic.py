"""Fig. 1: schematic of the model. The strategy frequency x evolves by replicator
dynamics with payoffs that depend on the resource level m; the two strategies harvest
the resource with efforts e_L and e_H."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "Arial"})
BLUE, RED, GREEN, GREY, ORANGE = "#2a6fb0", "#d0352f", "#2f9a3e", "#9c9c9c", "#e8801c"
BLUE_BG, GREEN_BG, INK, ARROW = "#eef4fb", "#eff8f0", "#222222", "#5a5a5a"
W, H = 2000, 1300

fig = plt.figure(figsize=(12, 12 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def Y(y):                                       # positions are measured from the top
    return H - y


def text(x, y, s, size, **kw):
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    return ax.text(x, Y(y), s, fontsize=size, **kw)


def width(s, size, **kw):
    """Width of a string in data units."""
    t = ax.text(0, 0, s, fontsize=size, **kw)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    inv = ax.transData.inverted()
    return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]


def box(x0, y0, x1, y1, edge, fill):
    ax.add_patch(FancyBboxPatch((x0, Y(y1)), x1 - x0, y1 - y0, lw=3, ec=edge, fc=fill,
                                boxstyle="round,pad=0,rounding_size=45"))


def bar(x0, y0, x1, y1, split, c_top, c_bottom, lab_top, lab_bottom, ink_top):
    ax.add_patch(Rectangle((x0, Y(y1)), x1 - x0, y1 - split, fc=c_bottom, ec="none"))
    ax.add_patch(Rectangle((x0, Y(split)), x1 - x0, split - y0, fc=c_top, ec="none"))
    ax.plot([x0, x1], [Y(split)] * 2, color=INK, lw=1.5)
    ax.add_patch(Rectangle((x0, Y(y1)), x1 - x0, y1 - y0, fc="none", ec=INK, lw=2))
    xc = (x0 + x1) / 2
    text(xc, (split + y1) / 2, lab_bottom, 30, ha="center", color="white")
    text(xc, (y0 + split) / 2, lab_top, 16, ha="center", color=ink_top)
    text(x0 - 30, y0, "1", 17, ha="center")
    text(x0 - 30, y1, "0", 17, ha="center")


def matrix(x, y, sub, size):
    h = 92
    for xs in ([x + 16, x, x, x + 16], [x + 204, x + 220, x + 220, x + 204]):
        ax.plot(xs, [Y(y - h), Y(y - h), Y(y + h), Y(y + h)], color=INK, lw=1.6)
    for i, row in enumerate(("RS", "TP")):
        for j, c in enumerate(row):
            text(x + 62 + 100 * j, y - 44 + 95 * i, f"${c}_{sub}$", size, ha="center")


# strategy evolution
box(300, 25, 1700, 425, BLUE, BLUE_BG)
bar(400, 75, 545, 375, 140, RED, BLUE, "$1-x$", "$x$", "white")
text(1130, 95, "EVOLUTION", 30, ha="center", fontweight="bold", color=BLUE)
text(1130, 200, "Replicator dynamics", 22, ha="center", fontweight="bold")
text(1130, 305, r"$\dot{x} = s x(1-x)\,(\pi_L(x,m)-\pi_H(x,m))$", 26, ha="center")

# environment
box(300, 875, 1700, 1275, GREEN, GREEN_BG)
bar(400, 925, 545, 1225, 1030, GREY, GREEN, "$1-m$", "$m$", "white")
text(1130, 945, "ENVIRONMENT", 30, ha="center", fontweight="bold", color=GREEN)
text(700, 1045, r"$\dot{m} = s\epsilon\,[(\lambda-\mu)\,m(1-m)$", 26)
text(830, 1120, r"$-\;qm\,(e_Lx+e_H(1-x))]$", 26)
text(830, 1197, r"$+\;\mathrm{stochasticity}$", 26, color=ORANGE)

# payoffs depend on the resource; weights coloured as the resource bar
text(160, 535, "Payoff matrix", 23, fontweight="bold")
x = 160
for item, col in ((r"$\Pi(m)=$", INK), (r"$(1-m)$", GREY), (0, None), ("$+$", INK), ("$m$", GREEN), (1, None)):
    if isinstance(item, int):
        matrix(x, 665, item, 23)
        x += 220 + 28
    else:
        text(x, 665, item, 24, color=col)
        x += width(item, 24) + 26
print(f"payoff formula ends at x = {x:.0f}")

# exploitation of the environment
text(1525, 590, "Exploitation of the", 22, ha="center", fontweight="bold")
text(1525, 655, "environment with", 22, ha="center", fontweight="bold")
pieces = [("harvest effort e", 22), ("L", 15), (" or e", 22), ("H", 15)]
widths = [width(s, size, fontweight="bold") for s, size in pieces]
x = 1525 - sum(widths) / 2
for (s, size), w in zip(pieces, widths):
    text(x, 720 + (12 if size < 20 else 0), s, size, fontweight="bold")
    x += w

# feedback arrows, routed outside the labels
style = "-|>,head_length=0.85,head_width=0.42"
for start, end in (((292, 1000), (292, 300)), ((1708, 300), (1708, 1000))):
    ax.add_patch(FancyArrowPatch((start[0], Y(start[1])), (end[0], Y(end[1])), arrowstyle=style,
                                 mutation_scale=30, connectionstyle="arc3,rad=-0.58",
                                 color=ARROW, lw=5, joinstyle="miter", capstyle="butt"))
fig.savefig("figures/fig1_schematic.png", dpi=200)
fig.savefig("figures/fig1_schematic.pdf")
