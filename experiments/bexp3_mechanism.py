"""bexp3: mechanism descriptors and their association with the history gain.

The descriptors are computed from source cells only (which keeps them legal) and
then correlated by Spearman with the test-side history gain from bexp1.
Positioning: an explanatory association; with n=11 it supports no predictive
claim.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.mechanism import compute_indicators

OUT = Path("results/battery/bexp3_mechanism.json")
BEXP1 = Path("results/battery/bexp1_author_protocol.jsonl")

INDICATORS = ["dim_row_ratio", "trend_strength", "mean_cell_len",
              "group_shift_auc", "random_minus_group", "n_source_rows"]


def history_gain(backbone: str = "tabpfn"):
    """Test-side gain of history over raw in per cent (negative favours history),
    cell_macro convention."""
    rows = [json.loads(l) for l in BEXP1.read_text().splitlines()]
    out = {}
    for uid, _, _ in FIT_UNITS:
        m = {}
        for view in ("raw", "history"):
            v = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
                 and r["view"] == view and r["backbone"] == backbone]
            if v:
                m[view] = float(np.mean(v))
        if len(m) == 2:
            out[uid] = (m["history"] - m["raw"]) / m["raw"] * 100
    return out


def main():
    if OUT.exists():
        data = json.load(OUT.open())
    else:
        data = {}
    for uid, ds, bk in FIT_UNITS:
        if uid in data:
            continue
        unit = load_fit_unit(ds, bk)
        data[uid] = compute_indicators(unit)
        json.dump(data, OUT.open("w"), indent=1)
        print(f"{uid:<14} ratio={data[uid]['dim_row_ratio']:.4f} "
              f"trend={data[uid]['trend_strength']:.3f} "
              f"auc={data[uid]['group_shift_auc']:.3f} "
              f"rnd-grp={data[uid]['random_minus_group']:+.3f}", flush=True)
    report(data)


def report(data=None):
    data = data or json.load(OUT.open())
    gains = history_gain("tabpfn")
    print("\n===== mechanism descriptors vs the history gain on TabPFN =====")
    hdr = f"{'unit':<15}{'gain%':>8}" + "".join(f"{k[:12]:>13}" for k in INDICATORS)
    print(hdr)
    for uid in data:
        if uid not in gains:
            continue
        line = f"{uid:<15}{gains[uid]:>+8.1f}"
        for k in INDICATORS:
            line += f"{data[uid][k]:>13.4f}"
        print(line)

    print("\n--- Spearman association (gain% vs descriptor, n={}) ---".format(
        len([u for u in data if u in gains])))
    for k in INDICATORS:
        xs = [data[u][k] for u in data if u in gains]
        ys = [gains[u] for u in data if u in gains]
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 4:
            continue
        r = spearmanr(np.array(xs)[ok], np.array(ys)[ok])
        print(f"  {k:<20} rho={r.statistic:+.3f}  p={r.pvalue:.3f}")
    print("\nnote: with n=11 the association is explanatory only and constitutes "
          "no predictive claim.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
