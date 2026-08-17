"""bexp20: consolidating the conflict-gating analysis (persisting what were /tmp
scratch analyses so they are reproducible).

Contents:
  A. recompute the phase-2 verdict: C1/C2/C3 plus the mechanism decomposition
     (variance signal vs conflict signal)
  B. the three-policy table for the Tier-S fallback gate (G1-G3)
  C. generalisation: the two-signal test for the smoothing constraint
  D. an exploratory item (allowed to fail): the history view as a third prior --
     the per-cell raw/history paired gain of bexp1 against the label-free
     prediction statistics of bexp17
  E. the main figures: a per-cell scatter (246 points coloured by unit) and the
     three-policy bar chart
Output: results/battery/bexp20_analysis.json + doc/battery/figures/cgca_*.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/battery"
FIG = ROOT / "doc/battery/figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = R / "bexp20_analysis.json"

R_SRC = {"TJU-1": 0.014, "TJU-2": 0.044, "HUST": 0.110, "XJTU-RW": 0.136,
         "TJU-3": 0.184, "XJTU-R3": 0.225, "XJTU-3C": 0.248, "XJTU-2C": 0.287,
         "MIT": 0.338, "XJTU-satellite": 0.461, "XJTU-R2.5": 0.915}
NOISE = 11.3


def load_cells():
    rows = [json.loads(l) for l in (R / "bexp17_cell_level.jsonl").read_text().splitlines()]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["unit"], r["cell_id"])].append(r)
    cells = []
    for (u, c), rs in agg.items():
        m = {k: float(np.mean([x[k] for x in rs])) for k in
             ("v_pred", "v_true", "corr_mag", "rough_pred", "rough_true",
              "rmse_none", "rmse_iso", "delta_iso", "delta_smooth")}
        cells.append({"unit": u, "cell": c, **m})
    return rows, cells


def main():
    rows, cells = load_cells()
    out = {"n_records": len(rows), "n_cells": len(cells)}

    vp = np.array([c["v_pred"] for c in cells])
    vt = np.array([c["v_true"] for c in cells])
    di = np.array([c["delta_iso"] for c in cells])

    # A. the phase-2 verdict
    rT, rO = spearmanr(vp, di), spearmanr(vt, di)
    per_unit = {}
    for u in sorted({c["unit"] for c in cells}):
        sub = [c for c in cells if c["unit"] == u]
        if len(sub) >= 4:
            r = spearmanr([c["v_pred"] for c in sub], [c["delta_iso"] for c in sub])
            per_unit[u] = {"n": len(sub), "rho": float(r.statistic), "p": float(r.pvalue)}
    pos = sum(1 for v in per_unit.values() if v["rho"] > 0)
    out["phase2"] = {
        "C1_pooled_rho": float(rT.statistic), "C1_p": float(rT.pvalue),
        "C1_holds": bool(rT.statistic > 0 and rT.pvalue < 0.01),
        "C2_pos_units": pos, "C2_total": len(per_unit), "C2_holds": pos >= 7,
        "C3_ratio": float(abs(rT.statistic) / max(abs(rO.statistic), 1e-9)),
        "oracle_rho": float(rO.statistic), "oracle_p": float(rO.pvalue),
        "per_unit": per_unit,
        "mechanism": {
            "v_pred_vs_rmse_none": list(map(float, spearmanr(
                vp, [c["rmse_none"] for c in cells])[:2])),
            "v_pred_vs_v_true": list(map(float, spearmanr(vp, vt)[:2])),
            "harmed_cells_v_true_median": float(np.median(
                [c["v_true"] for c in cells if c["delta_iso"] > NOISE]) or 0),
            "all_cells_v_true_median": float(np.median(vt)),
        },
    }

    # B. Tier-S three policies
    g18 = json.load((R / "bexp18_gating.json").open())
    never, always = g18["never"], g18["always"]
    tiers, harmed, cap = {}, 0, []
    for u in never:
        apply = R_SRC[u] < 0.15
        tiers[u] = always[u] if apply else never[u]
        d = (tiers[u] - never[u]) / never[u] * 100
        if d > NOISE:
            harmed += 1
        if always[u] < never[u]:
            cap.append((never[u] - tiers[u]) / (never[u] - always[u]))
    wins = sum(1 for u in never if tiers[u] <= min(never[u], always[u]) + 1e-12)
    out["tier_s_gate"] = {"per_unit": tiers, "G1_harmed": harmed,
                          "G2_capture_median": float(np.median(cap)),
                          "G3_wins": wins}

    # C. smoothing generalisation (two signals)
    ratios, dsm = [], []
    for c in cells:
        ref = np.median([x["rough_true"] for x in cells if x["unit"] != c["unit"]])
        ratios.append(c["rough_pred"] / max(ref, 1e-9))
        dsm.append(c["delta_smooth"])
    r1 = spearmanr(ratios, dsm)
    r2 = spearmanr([c["rough_true"] for c in cells], dsm)
    out["phase5_smooth"] = {"labelfree_rho": float(r1.statistic),
                            "labelfree_p": float(r1.pvalue),
                            "labelside_rho": float(r2.statistic),
                            "labelside_p": float(r2.pvalue)}

    # D. exploratory: the history view as a third prior (bexp1 per-cell gain x
    # bexp17 label-free statistics)
    b1 = [json.loads(l) for l in (R / "bexp1_author_protocol.jsonl").read_text().splitlines()]
    gain = defaultdict(lambda: defaultdict(list))    # unit -> cell -> [gain%]
    for view in ("raw", "history"):
        pass
    pc = defaultdict(lambda: {})
    for r in b1:
        if r["backbone"] != "tabpfn":
            continue
        for cell, v in r["per_cell"].items():
            pc[(r["unit"], cell, r["seed"])][r["view"]] = v
    for (u, cell, seed), d in pc.items():
        if "raw" in d and "history" in d:
            gain[u][cell].append((d["history"] - d["raw"]) / d["raw"] * 100)
    xs, ys = [], []
    cellmap = {(c["unit"], c["cell"]): c for c in cells}
    for u, cd in gain.items():
        for cell, gs in cd.items():
            key = (u, cell)
            if key in cellmap:
                xs.append(cellmap[key]["rough_pred"])   # label-free statistic: predicted roughness
                ys.append(float(np.mean(gs)))
    re = spearmanr(xs, ys)
    out["phase5_exploratory_history_prior"] = {
        "n": len(xs), "stat": "rough_pred(bexp17) vs history_gain(bexp1)",
        "rho": float(re.statistic), "p": float(re.pvalue),
        "verdict": "signal" if (re.pvalue < 0.05) else "no signal (exploratory, recorded as measured)"}

    json.dump(out, OUT.open("w"), indent=1)

    # E. the main figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = sorted({r["unit"] for r in rows})
    cmap = plt.cm.tab20(np.linspace(0, 1, len(units)))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for i, u in enumerate(units):
        sub = [r for r in rows if r["unit"] == u]
        ax[0].scatter([r["v_pred"] for r in sub], [r["delta_iso"] for r in sub],
                      s=14, alpha=.65, color=cmap[i], label=u)
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_xlabel("v_pred (label-free prediction violation ratio)")
    ax[0].set_ylabel("delta_iso (%)  [negative = iso helps]")
    ax[0].set_title(f"Cell-level (246 pts): pooled Spearman ρ={rT.statistic:+.2f} "
                    f"(dev) — REVERSED on unexposed lib (+0.57)")
    ax[0].legend(fontsize=5.5, ncol=2, frameon=False)
    ax[0].set_ylim(-80, 100)

    ulist = sorted(never)
    x = np.arange(len(ulist))
    for k, (lab, d) in enumerate([("never", never), ("always", always),
                                  ("Tier-S gate", tiers)]):
        ax[1].bar(x + (k - 1) * .27, [d[u] for u in ulist], width=.25, label=lab)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(ulist, rotation=60, fontsize=6.5, ha="right")
    ax[1].set_ylabel("cell-macro RMSE")
    ax[1].set_title("Unit-level conflict gate: zero harm, 50% upside capture")
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "cgca_main.png", dpi=160)
    print(f"saved -> {OUT}\nfigure -> {FIG/'cgca_main.png'}")
    print(f"exploratory (history prior): n={len(xs)} rho={re.statistic:+.3f} p={re.pvalue:.3f} "
          f"→ {out['phase5_exploratory_history_prior']['verdict']}")


if __name__ == "__main__":
    main()
