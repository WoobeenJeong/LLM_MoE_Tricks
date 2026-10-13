#!/usr/bin/env python
"""Draw docs/architecture.png - the one figure of this repo.

Everything is placed on an explicit grid (COL_*, ROW_Y) so the boxes line up by
construction instead of by eye.

  python docs/make_architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle

TRAIN = "#2d6cb3"       # trained in this run
FROZEN = "#b8b8b8"      # loaded and left alone
EDGE = "#333333"
BUS = "#7a7a7a"

# ---- grid ------------------------------------------------------------------
COL_IN = (0.4, 2.0)          # (x0, x1) of the input column
COL_EXP = (2.9, 7.1)
COL_MUL = 7.75               # centre of the multiply node
COL_CAT = (8.4, 9.3)
COL_CLF = (10.1, 11.9)
ROW_Y = [5.6, 3.95, 2.3]     # centre of each modality row
ROW_H = 1.35
GATE_Y = (0.30, 1.35)        # the gate block, in the expert column
BUS_IN = 2.45                # vertical bus: inputs -> gate
BUS_OUT = 7.42               # vertical bus: gate -> multiply nodes

# ---- per-row content: the point of the figure -------------------------------
ROWS = [
    dict(view="OMICS_01", arch="transformer", n_layers=4, trainable=1, w="w1",
         init="pretrained .pth   -   lr 1e-4",
         note="unfreeze_last_n: 1      last block adapts"),
    dict(view="OMICS_02", arch="mlp", n_layers=4, trainable=0, w="w2",
         init="pretrained .pth",
         note="unfreeze_last_n: 0      frozen, gate decides alone"),
    dict(view="OMICS_N", arch="cnn1d", n_layers=4, trainable=4, w="wN",
         init="from scratch   -   lr 3e-3",
         note="unfreeze_last_n: -1     everything trains"),
]

fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=200)
ax.set_xlim(0, 14.8)
ax.set_ylim(-0.1, 6.6)
ax.axis("off")


def box(x0, x1, yc, h, label, sub=None, fc="white", ec=EDGE, lw=1.2, fs=10):
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text((x0 + x1) / 2, yc + (0.17 * h if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=EDGE)
    if sub:
        ax.text((x0 + x1) / 2, yc - 0.26 * h, sub, ha="center", va="center",
                fontsize=fs - 2, color="#555555")


def arrow(x0, y0, x1, y1, style="-", color=EDGE, lw=1.2, head=0.12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw,
                                linestyle=style, shrinkA=0, shrinkB=0,
                                mutation_scale=12))


def line(x0, y0, x1, y1, color=BUS, lw=1.1, style="--"):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, linestyle=style,
            solid_capstyle="butt", zorder=0)


# ---- modality rows ----------------------------------------------------------
for row, yc in zip(ROWS, ROW_Y):
    # input
    box(*COL_IN, yc, 0.72, row["view"], sub="n x d", fc="#f4f4f4")

    # expert: one small block per layer, grey = frozen, blue = trained
    x0, x1 = COL_EXP
    ax.add_patch(FancyBboxPatch((x0, yc - ROW_H / 2), x1 - x0, ROW_H,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=EDGE, linewidth=1.2))
    ax.text(x0 + 0.22, yc + 0.44, f"expert : {row['arch']}", ha="left",
            va="center", fontsize=10, color=EDGE)
    ax.text(x1 - 0.22, yc + 0.44, row["init"], ha="right", va="center",
            fontsize=8.5, color="#555555", style="italic")

    n = row["n_layers"]
    bw, gap = 0.58, 0.14
    bx, by = x0 + 0.22, yc - 0.10
    for i in range(n):
        trained = i >= n - row["trainable"]
        ax.add_patch(Rectangle((bx + i * (bw + gap), by), bw, 0.30,
                               facecolor=TRAIN if trained else FROZEN,
                               edgecolor="none"))
    ax.text(bx + n * (bw + gap) + 0.10, by + 0.15, "-> h", ha="left",
            va="center", fontsize=9, color="#555555")
    ax.text(x0 + 0.22, yc - 0.46, row["note"], ha="left", va="center",
            fontsize=8.5, color=EDGE)

    arrow(COL_IN[1], yc, x0, yc)

    # gate-weighted representation
    ax.add_patch(plt.Circle((COL_MUL, yc), 0.17, facecolor="white",
                            edgecolor=EDGE, linewidth=1.2))
    ax.text(COL_MUL, yc, "\u00d7", ha="center", va="center", fontsize=11,
            color=EDGE)
    arrow(x1, yc, COL_MUL - 0.17, yc)
    arrow(COL_MUL + 0.17, yc, COL_CAT[0], yc)

    # raw view -> gate (dashed bus down the left of the expert column)
    line(COL_IN[1], yc, BUS_IN, yc)
    line(BUS_IN, yc, BUS_IN, (GATE_Y[0] + GATE_Y[1]) / 2)

# ---- gate -------------------------------------------------------------------
gy = (GATE_Y[0] + GATE_Y[1]) / 2
box(COL_EXP[0], COL_EXP[1], gy, GATE_Y[1] - GATE_Y[0],
    "GATE :  sigmoid | softmax | gumbel | attention",
    sub="always trained from scratch  -  learned temperature  -  train.lr",
    fc="#eaf1f9", ec=TRAIN, lw=1.5, fs=10)
arrow(BUS_IN, gy, COL_EXP[0], gy, style="--", color=BUS)
ax.text(BUS_IN, ROW_Y[0] + ROW_H / 2 + 0.12, "raw views", ha="center",
        va="bottom", fontsize=8.5, color=BUS)

# gate -> each multiply node, entering from below so it never runs along the
# expert -> multiply arrow
line(COL_EXP[1], gy, BUS_OUT, gy, color=TRAIN, style="-")
for row, yc in zip(ROWS, ROW_Y):
    line(BUS_OUT, gy, BUS_OUT, yc - 0.45, color=TRAIN, style="-")
    line(BUS_OUT, yc - 0.45, COL_MUL, yc - 0.45, color=TRAIN, style="-")
    arrow(COL_MUL, yc - 0.45, COL_MUL, yc - 0.17, color=TRAIN, lw=1.1)
    ax.text(COL_MUL + 0.12, yc - 0.36, row["w"], ha="left", va="center",
            fontsize=9, color=TRAIN)

# ---- joint representation -> classifier ------------------------------------
cat_yc = (ROW_Y[0] + ROW_Y[-1]) / 2
cat_h = ROW_Y[0] - ROW_Y[-1] + ROW_H
box(COL_CAT[0], COL_CAT[1], cat_yc, cat_h, "", fc="#f4f4f4")
ax.text((COL_CAT[0] + COL_CAT[1]) / 2, cat_yc, "concat", ha="center",
        va="center", fontsize=10, rotation=90, color=EDGE)

box(COL_CLF[0], COL_CLF[1], cat_yc, 0.8, "linear classifier", fc="#eaf1f9",
    ec=TRAIN, lw=1.5)
arrow(COL_CAT[1], cat_yc, COL_CLF[0], cat_yc)
arrow(COL_CLF[1], cat_yc, COL_CLF[1] + 0.55, cat_yc)
ax.text(COL_CLF[1] + 0.65, cat_yc, "logit", ha="left", va="center", fontsize=10,
        color=EDGE)

# ---- where each loss attaches ----------------------------------------------
ax.text(COL_CLF[0], cat_yc + 0.75, "task loss", ha="left", va="bottom",
        fontsize=9, color=EDGE, weight="bold")
ax.text(COL_CLF[0], cat_yc + 0.45, "bce | wbce | focal | triplet", ha="left",
        va="bottom", fontsize=8.5, color="#555555")

ax.text(COL_CAT[0], cat_yc - cat_h / 2 - 0.30, "representation loss",
        ha="left", va="top", fontsize=9, color=EDGE, weight="bold")
ax.text(COL_CAT[0], cat_yc - cat_h / 2 - 0.58,
        "over replicate pairs\nntxent | barlow_twins | vicreg", ha="left",
        va="top", fontsize=8.5, color="#555555")

ax.text(COL_EXP[0], GATE_Y[0] - 0.12, "gate loss", ha="left", va="top",
        fontsize=9, color=EDGE, weight="bold")
ax.text(COL_EXP[0] + 1.05, GATE_Y[0] - 0.12,
        "auxiliary | importance | entropy", ha="left", va="top", fontsize=8.5,
        color="#555555")

# ---- legend -----------------------------------------------------------------
lx, ly = COL_IN[0], GATE_Y[0] + 0.62
ax.add_patch(Rectangle((lx, ly), 0.34, 0.22, facecolor=TRAIN, edgecolor="none"))
ax.text(lx + 0.44, ly + 0.11, "trained", va="center", fontsize=8.5, color=EDGE)
ax.add_patch(Rectangle((lx, ly - 0.42), 0.34, 0.22, facecolor=FROZEN,
                       edgecolor="none"))
ax.text(lx + 0.44, ly - 0.31, "frozen", va="center", fontsize=8.5, color=EDGE)

fig.tight_layout(pad=0.4)
out = __file__.replace("make_architecture.py", "architecture.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
