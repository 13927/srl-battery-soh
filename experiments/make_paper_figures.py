"""Generate Figure 1-4 for the Energies manuscript at 300 dpi.

Replaces an earlier throwaway script whose point labels used hand-tuned fixed
offsets; the same offset table was reused for two panels with different y-axes,
so labels collided. Labels are now placed by a small collision-avoidance search
(`place_labels`) that scores candidate positions against already-placed labels,
the data points, and the axes border, and draws a leader line when the label has
to sit far from its point.

Usage: .venv/bin/python experiments/make_paper_figures.py
Outputs: doc/battery/figures/Figure{1,2,3,4}.png
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from scipy.stats import spearmanr

from battery_lab.data_adapters import FIT_UNITS

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

UNITS = [u for u, _, _ in FIT_UNITS]
FIGDIR = ROOT / "doc/battery/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results/battery"
BLUE, RED, ORANGE, GREY = "#2e5f8a", "#b03a2e", "#b0651e", "#9aa7b3"


def _overlap(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


# Clearance around a label box. Generous on purpose: boxes that merely touch are
# mathematically disjoint but read as one run of text, which is exactly how the
# previous version of this figure failed.
PAD_X, PAD_Y = 1.22, 1.95
MARKER_CLEAR = 5.0   # px kept free around every data point


def place_labels(ax, xs, ys, labels, fontsize=6.5, leader_from=15.0, sweeps=3,
                 extra=()):
    """Annotate points, choosing offsets that avoid labels, points and borders.

    Greedy first pass followed by refinement sweeps, so the result does not
    depend on the order in which the units happen to be listed. `extra` holds
    artists already drawn in the axes (e.g. region annotations) that labels must
    also keep clear of.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0

    pts = ax.transData.transform(np.column_stack([xs, ys]))
    axes_bb = ax.get_window_extent(rend)
    marks = [Bbox.from_bounds(p[0] - MARKER_CLEAR, p[1] - MARKER_CLEAR,
                              2 * MARKER_CLEAR, 2 * MARKER_CLEAR) for p in pts]
    marks += [a.get_window_extent(renderer=rend).expanded(1.1, 1.3) for a in extra]

    # candidate offsets in points: rings of increasing distance
    cands = []
    for dist in (7, 11, 16, 22, 30, 40, 52):
        for ang in range(0, 360, 20):
            cands.append((dist * np.cos(np.radians(ang)), dist * np.sin(np.radians(ang))))

    # measure every label once
    sizes = []
    for (x, y), lab in zip(zip(xs, ys), labels):
        probe = ax.annotate(lab, (x, y), xytext=(0, 0), textcoords="offset points",
                            fontsize=fontsize, ha="left", va="center")
        ext = probe.get_window_extent(renderer=rend)
        sizes.append((ext.width, ext.height))
        probe.remove()

    def box(i, dx, dy):
        w, h = sizes[i]
        ddx, ddy = dx * px_per_pt, dy * px_per_pt
        x0 = pts[i][0] + ddx if ddx >= 0 else pts[i][0] + ddx - w
        return Bbox.from_bounds(x0, pts[i][1] + ddy - h / 2, w, h).expanded(PAD_X, PAD_Y)

    def cost(i, dx, dy, boxes):
        bb = box(i, dx, dy)
        c = sum(_overlap(bb, boxes[j]) for j in range(len(boxes)) if j != i and boxes[j])
        c += 2.0 * sum(_overlap(bb, m) for m in marks)
        c += 30.0 * (max(0.0, axes_bb.x0 - bb.x0) + max(0.0, bb.x1 - axes_bb.x1)
                     + max(0.0, axes_bb.y0 - bb.y0) + max(0.0, bb.y1 - axes_bb.y1))
        c += 0.35 * (dx * dx + dy * dy) ** 0.5     # prefer labels close to their point
        c += 0.0 if dx >= 0 else 2.0               # mild right-side preference
        return c

    chosen = [None] * len(labels)
    boxes = [None] * len(labels)
    order = sorted(range(len(labels)), key=lambda i: -sizes[i][0])   # widest first
    for _ in range(sweeps):
        for i in order:
            best = min(cands, key=lambda c: cost(i, c[0], c[1], boxes))
            chosen[i] = best
            boxes[i] = box(i, *best)

    for i, lab in enumerate(labels):
        dx, dy = chosen[i]
        cx, cy = (boxes[i].x0 + boxes[i].x1) / 2, (boxes[i].y0 + boxes[i].y1) / 2
        d_own = ((cx - pts[i][0]) ** 2 + (cy - pts[i][1]) ** 2) ** 0.5
        d_other = min(((cx - p[0]) ** 2 + (cy - p[1]) ** 2) ** 0.5
                      for j, p in enumerate(pts) if j != i)
        # A leader line is drawn whenever the label is far from its point, or when
        # some other point is not clearly farther away — otherwise a reader could
        # attribute the label to the wrong unit.
        ambiguous = d_other < 1.25 * d_own
        kw = {}
        if (dx * dx + dy * dy) ** 0.5 >= leader_from or ambiguous:
            kw["arrowprops"] = dict(arrowstyle="-", color="0.45", lw=0.4,
                                    shrinkA=1.0, shrinkB=2.0)
        ax.annotate(lab, (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, ha="left" if dx >= 0 else "right",
                    va="center", color="0.15", **kw)

    worst = max((_overlap(boxes[i], boxes[j]) for i in range(len(boxes))
                 for j in range(i + 1, len(boxes))), default=0.0)
    print(f"  {ax.get_title()[:34]!r}: worst label-label overlap = {worst:.1f} px^2")
    return worst


# W04: the importance source is now the **ten-seed mean** (bexp52), the same source as
# Table 2's column and Figure 1b's rho. Build a view compatible with the old p6
# structure (_{unit: {family: mean}}) so downstream consumers are unchanged.
_B52 = json.load(open(RES / "bexp52_importance.json"))["per_unit"]
_FAMS = ["raw", "cycle", "lag1", "diff1", "from_first", "roll5"]
imp = {u: {f: fams[f]["mean"] for f in _FAMS if f in fams}
       for u, fams in _B52.items()}
imp_sd = {u: {f: fams[f]["std"] for f in _FAMS if f in fams}
          for u, fams in _B52.items()}
# the old single-point archive is kept as a historical record (reference only, no longer plotted): p6_importance.json
cyc = [imp[u]["cycle"] for u in UNITS]
ff = [imp[u]["from_first"] for u in UNITS]

# ---------------------------------------------------------------- Figure 1 ----
d = json.load(open(RES / "bexp42_cycle_convention.json"))
sens = [(d[f"{u}|deployable"]["rmse"] - d[f"{u}|lifetime"]["rmse"])
        / d[f"{u}|lifetime"]["rmse"] * 100 for u in UNITS]
r = spearmanr(cyc, sens)

fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.3))
order = np.argsort(sens)[::-1]
ax[0].barh([UNITS[i] for i in order], [sens[i] for i in order],
           color=[RED if sens[i] > 0 else BLUE for i in order], edgecolor="k", lw=.5)
ax[0].axvline(0, color="k", lw=.8)
ax[0].set_xlabel("Change in RMSE (%)")
ax[0].set_title("(a) Effect of making the cycle column deployable")

ax[1].scatter(cyc, sens, s=42, color=BLUE, edgecolor="k", lw=.4, zorder=3)
ax[1].axhline(0, color="k", lw=.8)
ax[1].set_xlabel("Life-fraction cycle importance (%)")
ax[1].set_ylabel("Change in RMSE (%)")
ax[1].set_title(f"(b) Spearman $\\rho$={r.statistic:+.2f}, p={r.pvalue:.2f} (n=11)")
ax[1].set_xlim(-10, 118)
ax[1].margins(y=0.16)
fig.tight_layout()
place_labels(ax[1], cyc, sens, UNITS)
fig.savefig(FIGDIR / "Figure1.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2 ----
sig = json.load(open(RES / "bexp36_significance.json"))
fig, ax = plt.subplots(figsize=(7.0, 4.6))
y = np.arange(len(UNITS))[::-1]
for off, bk, col, mk, lab in ((0.17, "tabpfn", BLUE, "o", "TabPFN (full stack)"),
                              (-0.17, "gbdt", ORANGE, "s", "GBDT (full stack)")):
    for i, u in enumerate(UNITS):
        rr = sig[f"{bk}|{u}"]
        yy = y[i] + off
        lo, hi = rr["ci95"]
        dd = rr["delta_pct"]
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.4)
        ax.plot(dd, yy, mk, color=col, ms=4.6, mec="k", mew=.4,
                label=lab if i == 0 else None)
        if rr["p_holm"] < 0.05 and dd < 0:
            ax.text(hi + 1.4, yy, "*", color=col, fontsize=10, va="center")
ax.axvline(0, color="k", lw=1)
ax.set_yticks(y, UNITS)
ax.set_xlabel("Relative RMSE versus the physics-informed baseline (%)")
ax.set_title("Per-unit comparison under the reference protocol\n"
             "(bars: bootstrap 95% CI; *: Holm-corrected p < 0.05)")
ax.legend(frameon=False, loc="lower left")
ax.grid(axis="x", alpha=.2)
fig.tight_layout()
fig.savefig(FIGDIR / "Figure2.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3 ----
s = json.load(open(RES / "bexp27_randomization.json"))
T = np.zeros(100)
Ts = 0.0
for u in UNITS:
    T += np.array([s[u][f"perm{m}"] for m in range(1, 101)]) / len(UNITS)
    Ts += s[u]["self"] / len(UNITS)

fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.3))
counts, _, _ = ax[0].hist(T, bins=16, color=GREY, edgecolor="k", lw=.5,
                          label="100 mispaired references")
ax[0].axvline(Ts, color=RED, lw=2.2, label="True self-pairing")
ax[0].annotate("p = 0.0099", xy=(Ts, 7), xytext=(Ts + 0.0007, 8), fontsize=8,
               color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=.8))
ax[0].set_ylim(0, counts.max() * 1.32)   # headroom so the legend clears the bars
ax[0].set_xlabel("Mean RMSE across 11 evaluation units")
ax[0].set_ylabel("Count")
ax[0].set_title("(a) Randomization inference")
ax[0].legend(frameon=False, loc="upper left")

ax[1].axhspan(50, 100, color=BLUE, alpha=.07)
ax[1].axvspan(50, 100, color=ORANGE, alpha=.07)
# placed in the empty interior of each band, and passed to place_labels below so
# that unit labels keep clear of them
band_labels = [
    ax[1].text(6, 62, "Online carrier\ndominant", color=BLUE, fontsize=7.5),
    ax[1].text(54, 40, "Offline carrier\ndominant", color=ORANGE, fontsize=7.5),
]
ax[1].scatter(cyc, ff, s=46, color=BLUE, edgecolor="k", lw=.4, zorder=3)
ax[1].set_xlabel("Life-fraction cycle importance (%)")
ax[1].set_ylabel("Offset feature importance (%)")
ax[1].set_title("(b) Two carriers of the same information")
ax[1].set_xlim(-10, 118)
ax[1].set_ylim(-10, 108)
fig.tight_layout()
place_labels(ax[1], cyc, ff, UNITS, extra=band_labels)
fig.savefig(FIGDIR / "Figure3.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Figure 4 ----
# The two probes reported in Section 3.9 and the Limitations. Panel (b) uses the
# 20-seed stage-2 run, not the 5-seed judgment run: with 20 seeds the learned
# anchor is significantly worse on 6 of 11 units, which the 5-seed run could not
# resolve.
A = json.load(open(RES / "bexp45_learnable_ref.json"))
A2 = json.load(open(RES / "bexp45_stage2.json"))
C = json.load(open(RES / "bexp46_distill.json"))

from scipy.stats import mannwhitneyu

fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.3))

# (a) where the learned anchor draws from
alpha = np.array([A[u]["alpha_mean"] for u in UNITS])
k = alpha.shape[1]
pos = np.arange(1, k + 1)
for row in alpha:
    ax[0].plot(pos, row, color=GREY, lw=.8, alpha=.75, zorder=2)
ax[0].plot(pos, alpha.mean(0), color=BLUE, lw=2.2, marker="o", ms=4, zorder=4,
           label="Mean over 11 units")
ax[0].axhline(1 / k, color=RED, ls="--", lw=1.2, zorder=3,
              label=f"Uniform ({1/k:.2f})")
ax[0].set_xticks(pos)
ax[0].set_xlabel("Cycle index within the early window")
ax[0].set_ylabel("Learned attention weight")
ax[0].set_title("(a) Where the learned anchor draws from")
ax[0].set_ylim(0, max(alpha.max() * 1.35, 1 / k * 2.2))
ax[0].legend(frameon=False, loc="upper right")

# (b) per-unit penalty of the learned anchor, 20 seeds, Holm-corrected
rows = [(u, A2[f"{u}|gbdt"]) for u in UNITS if f"{u}|gbdt" in A2]
deltas = [(r["learnable_ref"]["rmse"] - r["fixed_srl"]["rmse"])
          / r["fixed_srl"]["rmse"] * 100 for _, r in rows]
raw_p = [float(mannwhitneyu(r["learnable_ref"]["seed_vals"],
                            r["fixed_srl"]["seed_vals"],
                            alternative="two-sided").pvalue) for _, r in rows]
order = sorted(range(len(raw_p)), key=lambda i: raw_p[i])
holm, run_max = [0.0] * len(raw_p), 0.0
for rank, i in enumerate(order):
    run_max = max(run_max, raw_p[i] * (len(raw_p) - rank))
    holm[i] = min(1.0, run_max)
idx = np.argsort(deltas)
yy = np.arange(len(idx))
cols = [RED if (holm[i] < .05 and deltas[i] > 0) else
        BLUE if (holm[i] < .05 and deltas[i] < 0) else GREY for i in idx]
ax[1].barh(yy, [deltas[i] for i in idx], color=cols, edgecolor="k", lw=.4)
ax[1].axvline(0, color="k", lw=.9)
ax[1].set_yticks(yy)
ax[1].set_yticklabels([rows[i][0] for i in idx], fontsize=7)
ax[1].set_xlabel("Learned anchor vs fixed first cycle (%)")
ax[1].set_title("(b) The fixed first cycle is better")
for j, i in enumerate(idx):
    if holm[i] < .05:
        d = deltas[i]
        ax[1].text(d + (0.6 if d > 0 else -0.6), j, "*", va="center",
                   ha="left" if d > 0 else "right", fontsize=10,
                   color=RED if d > 0 else BLUE)
ax[1].margins(x=.12)                      # room for the significance markers
ax[1].text(.03, .06, "* Holm-corrected $p<0.05$", transform=ax[1].transAxes,
           ha="left", fontsize=7, color="0.3")

# (c) distillation dose-response
lams = [0.3, 0.5, 0.7]
per_lam = [[(C[u][f"distilled{l}"]["rmse"] - C[u]["online_only"]["rmse"])
            / C[u]["online_only"]["rmse"] * 100 for u in UNITS] for l in lams]
for j in range(len(UNITS)):
    ax[2].plot(lams, [per_lam[i][j] for i in range(3)], color=GREY, lw=.8,
               alpha=.75, marker=".", ms=3, zorder=2)
ax[2].plot(lams, [np.median(v) for v in per_lam], color=ORANGE, lw=2.2,
           marker="s", ms=5, zorder=4, label="Median over 11 units")
ax[2].axhline(0, color="k", lw=.9, zorder=3)
ax[2].axhspan(-11.3, 11.3, color=GREY, alpha=.18, zorder=1,
              label="Seed noise floor")
ax[2].set_xticks(lams)
ax[2].set_xlabel("Distillation weight $\\lambda$")
ax[2].set_ylabel("Change vs no distillation (%)")
ax[2].set_title("(c) The offline carrier does not transfer")
ax[2].legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(FIGDIR / "Figure4.png", dpi=300)
plt.close(fig)
print(f"Figure4: learned anchor significantly worse on "
      f"{sum(h < .05 and d > 0 for h, d in zip(holm, deltas))}/11 units, "
      f"better on {sum(h < .05 and d < 0 for h, d in zip(holm, deltas))}/11")

print("Figure1-4 written to", FIGDIR)
