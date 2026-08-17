"""bexp4: what abstention costs, and a post-hoc diagnosis of the decision rule.

The retrospective check gave 3 hits / 8 abstentions / 0 misses. Two things are
quantified here:
  1. what abstaining (averaging the two views) costs relative to the post-hoc best
     view and to a fixed single view
     -- the per-cell predictions of bexp1 cannot be averaged directly, so a
        conservative upper bound on the RMSE is used: the RMSE of the averaged
        prediction is at most the arithmetic mean of the two RMSEs (Jensen), and
        that bound is reported against the actual best and worst
  2. why abstention is so frequent: the ratio of CI width to effect size (a
     diagnosis only; no threshold is changed)

To be explicit: the thresholds are frozen, and this analysis feeds the discussion
rather than any tuning.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS

BEXP1 = Path("results/battery/bexp1_author_protocol.jsonl")
BEXP2 = Path("results/battery/bexp2_view_selection.jsonl")
OUT = Path("results/battery/bexp4_abstain_analysis.json")


def main():
    r1 = [json.loads(l) for l in BEXP1.read_text().splitlines()]
    r2 = [json.loads(l) for l in BEXP2.read_text().splitlines()]

    rows = []
    for uid, _, _ in FIT_UNITS:
        m = {}
        for view in ("raw", "history"):
            v = [r["rmse_cell_macro"] for r in r1 if r["unit"] == uid
                 and r["view"] == view and r["backbone"] == "tabpfn"]
            m[view] = float(np.mean(v))
        votes = [r["decision"] for r in r2 if r["unit"] == uid]
        counts = {v: votes.count(v) for v in ("raw", "history", "abstain")}
        top = max(counts.values())
        cand = [v for v, c in counts.items() if c == top]
        final = cand[0] if len(cand) == 1 else "abstain"

        best_view = "history" if m["history"] < m["raw"] else "raw"
        best, worst = min(m.values()), max(m.values())
        avg_upper = (m["raw"] + m["history"]) / 2   # conservative bound for abstain

        # cost of the three policies against the post-hoc best, in per cent
        cost_ruleD = (avg_upper if final == "abstain" else m[final])
        # source-side diagnosis: |effect| / half-width of the CI
        eff, halfw = [], []
        for r in r2:
            if r["unit"] != uid:
                continue
            lo, hi = r["ci"]
            eff.append(abs(r["mean_diff"]))
            halfw.append((hi - lo) / 2)
        snr = float(np.mean(eff) / np.mean(halfw)) if halfw else float("nan")

        rows.append({
            "unit": uid, "decision": final, "best_view": best_view,
            "rmse_raw": m["raw"], "rmse_history": m["history"],
            "rmse_best": best, "rmse_worst": worst,
            "rmse_ruleD_bound": cost_ruleD,
            "cost_vs_best_pct": (cost_ruleD - best) / best * 100,
            "gain_vs_worst_pct": (worst - cost_ruleD) / worst * 100,
            "source_snr": snr,
            "n_source_cells": [r["n_cells"] for r in r2
                               if r["unit"] == uid][0],
        })

    json.dump(rows, OUT.open("w"), indent=1)
    report(rows)


def report(rows=None):
    rows = rows or json.load(OUT.open())
    print("===== cost of the decision rule (cell_macro, reference protocol, "
          "mean of 5 seeds) =====")
    print(f"{'unit':<15}{'decision':<10}{'best':<9}{'raw':>9}{'hist':>9}"
          f"{'bound':>10}{'vs best%':>10}{'vs worst%':>11}{'SNR':>7}"
          f"{'cells':>7}")
    for r in rows:
        print(f"{r['unit']:<15}{r['decision']:<9}{r['best_view']:<9}"
              f"{r['rmse_raw']:>9.6f}{r['rmse_history']:>9.6f}"
              f"{r['rmse_ruleD_bound']:>10.6f}{r['cost_vs_best_pct']:>+9.2f}"
              f"{r['gain_vs_worst_pct']:>+9.2f}{r['source_snr']:>9.2f}"
              f"{r['n_source_cells']:>7}")

    print("\n--- summary over 11 units ---")
    for key, label in [("cost_vs_best_pct", "rule vs post-hoc best (smaller is better)"),
                       ("gain_vs_worst_pct", "rule vs post-hoc worst (larger is better)")]:
        v = [r[key] for r in rows]
        print(f"  {label}: mean {np.mean(v):+.2f}%  median {np.median(v):+.2f}%  "
              f"worst {max(v) if key.startswith('cost') else min(v):+.2f}%")

    # fixed single-view policies, for contrast
    always_h = [(r["rmse_history"] - r["rmse_best"]) / r["rmse_best"] * 100
                for r in rows]
    always_r = [(r["rmse_raw"] - r["rmse_best"]) / r["rmse_best"] * 100
                for r in rows]
    ruleD = [r["cost_vs_best_pct"] for r in rows]
    print(f"\n--- mean cost of the three policies against the post-hoc best ---")
    print(f"  always history : {np.mean(always_h):+.2f}%  "
          f"(worst unit {max(always_h):+.2f}%)")
    print(f"  always raw     : {np.mean(always_r):+.2f}%  "
          f"(worst unit {max(always_r):+.2f}%)")
    print(f"  the rule       : {np.mean(ruleD):+.2f}%  "
          f"(worst unit {max(ruleD):+.2f}%)")
    n_miss = sum(1 for r in rows if r["decision"] not in
                 ("abstain", r["best_view"]))
    print(f"\n  times the rule chose the wrong direction: {n_miss}/11")
    print(f"  units with SNR<1 (CI wider than the effect, the main reason for "
          f"abstention): "
          f"{sum(1 for r in rows if r['source_snr'] < 1)}/11")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
