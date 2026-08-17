"""bexp36: significance of the head-to-head comparison (analysis only, no
pass/fail line).

Opponent: the official PINN4SOH xlsx, per-experiment battery_mean RMSE (10 values
per unit).
Ours: bexp1 (TabPFN + history) and bexp14 (GBDT + history, context cap 2048),
cell-macro RMSE per seed.
Per unit: two-sided Mann-Whitney U, Holm correction over the 11 tests, and a
bootstrap 95 per cent CI of the relative difference.
Output: the significance columns of the manuscript table
(bexp36_significance.json).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

OUT = ROOT / "results/battery/bexp36_significance.json"
XL = ROOT / "external/PINN4SOH/results analysis/processed results"
UNIT_SHEET = [  # (unit, xlsx library, sheet/batch)
    ("XJTU-2C", "XJTU", 0), ("XJTU-3C", "XJTU", 1), ("XJTU-R2.5", "XJTU", 2),
    ("XJTU-R3", "XJTU", 3), ("XJTU-RW", "XJTU", 4), ("XJTU-satellite", "XJTU", 5),
    ("TJU-1", "TJU", 0), ("TJU-2", "TJU", 1), ("TJU-3", "TJU", 2),
    ("HUST", "HUST", 0), ("MIT", "MIT", 0),
]


def ours_per_seed(unit, backbone):
    if backbone == "tabpfn":
        rows = [json.loads(l) for l in
                open(ROOT / "results/battery/bexp1_author_protocol.jsonl")]
        vals = [r["rmse_cell_macro"] for r in rows
                if r["unit"] == unit and r["view"] == "history"
                and r["backbone"] == "tabpfn"]
    else:
        rows = [json.loads(l) for l in
                open(ROOT / "results/battery/bexp14_gbdt.jsonl")]
        vals = [r["rmse_cell_macro"] for r in rows
                if r["unit"] == unit and r["view"] == "history"
                and r.get("cap") == 2048]
    return vals


def main():
    res = {}
    pvals = {}
    for backbone in ("tabpfn", "gbdt"):
        for unit, lib, sh in UNIT_SHEET:
            theirs = pd.read_excel(XL / f"Ours-{lib}-results.xlsx",
                                   sheet_name=f"battery_mean_{sh}")["RMSE"].tolist()
            ours = ours_per_seed(unit, backbone)
            mw = mannwhitneyu(ours, theirs, alternative="two-sided")
            delta = (np.mean(ours) - np.mean(theirs)) / np.mean(theirs) * 100
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(5000):
                o = rng.choice(ours, len(ours)); t = rng.choice(theirs, len(theirs))
                boots.append((np.mean(o) - np.mean(t)) / np.mean(t) * 100)
            key = f"{backbone}|{unit}"
            res[key] = {
                "ours_mean": float(np.mean(ours)), "theirs_mean": float(np.mean(theirs)),
                "n_ours": len(ours), "n_theirs": len(theirs),
                "delta_pct": float(delta),
                "ci95": [float(np.percentile(boots, 2.5)),
                         float(np.percentile(boots, 97.5))],
                "p_raw": float(mw.pvalue),
            }
            pvals[key] = mw.pvalue
    # Holm correction, applied to the 11 tests of each backbone separately
    for backbone in ("tabpfn", "gbdt"):
        keys = [k for k in pvals if k.startswith(backbone)]
        order = sorted(keys, key=lambda k: pvals[k])
        m = len(order)
        prev = 0.0
        for i, k in enumerate(order):
            adj = min(1.0, max(prev, (m - i) * pvals[k]))
            res[k]["p_holm"] = float(adj)
            prev = adj
    json.dump(res, OUT.open("w"), indent=1)

    for backbone in ("tabpfn", "gbdt"):
        print(f"\n{'='*86}\n{backbone.upper()} + history vs the published PINN "
              f"(per unit, Holm corrected)")
        print(f"{'unit':<16}{'ours':>9}{'published':>11}{'delta%':>9}"
              f"{'CI95':>20}{'p':>9}{'p_holm':>9}")
        sig = 0
        for unit, _, _ in UNIT_SHEET:
            r = res[f"{backbone}|{unit}"]
            star = "✓" if r["p_holm"] < 0.05 and r["delta_pct"] < 0 else ""
            if star:
                sig += 1
            print(f"{unit:<16}{r['ours_mean']:>9.5f}{r['theirs_mean']:>9.5f}"
                  f"{r['delta_pct']:>+8.1f}"
                  f"  [{r['ci95'][0]:+6.1f},{r['ci95'][1]:+6.1f}]"
                  f"{r['p_raw']:>9.4f}{r['p_holm']:>9.4f} {star}")
        print(f"significantly better than published "
              f"(Holm p<0.05 and delta<0): {sig}/11")


if __name__ == "__main__":
    main()
