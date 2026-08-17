"""bexp9: path dependence -- do cross-cycle features separate contradictory
neighbours, and does that track the gain?

Neighbour SOH discordance and the snapshot-twin rate are measured in the raw and
history spaces over the 11 units, then correlated with the history-versus-raw
TabPFN gain of bexp1: if separating more contradictory neighbours goes with a
larger gain, the path-dependence account is supported. A pure kNN measurement,
seconds to run, TabPFN is not loaded.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.path_dependence import analyze_unit

OUT = Path("results/battery/bexp9_path_dependence.json")
BEXP1 = Path("results/battery/bexp1_author_protocol.jsonl")


def history_gain():
    rows = [json.loads(l) for l in BEXP1.read_text().splitlines()]
    g = {}
    for uid, _, _ in FIT_UNITS:
        m = {}
        for v in ("raw", "history"):
            vals = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
                    and r["view"] == v and r["backbone"] == "tabpfn"]
            if vals:
                m[v] = np.mean(vals)
        if len(m) == 2:
            g[uid] = (m["history"] - m["raw"]) / m["raw"] * 100
    return g


def main():
    data = json.load(OUT.open()) if OUT.exists() else {}
    for uid, ds, bk in FIT_UNITS:
        if uid in data:
            continue
        unit = load_fit_unit(ds, bk)
        data[uid] = analyze_unit(unit)
        json.dump(data, OUT.open("w"), indent=1)
        r, h = data[uid]["raw"], data[uid]["history"]
        print(f"{uid:<15} nbr_gap raw={r['mean_neighbor_soh_gap']:.4f} "
              f"hist={h['mean_neighbor_soh_gap']:.4f} "
              f"({data[uid]['gap_reduction_pct']:+.1f}%)  "
              f"twin raw={r['twin_rate']:.3f} hist={h['twin_rate']:.3f}",
              flush=True)
    report(data)


def report(data=None):
    data = data or json.load(OUT.open())
    gains = history_gain()
    print("\n===== path-dependence vs History@TabPFN gain =====")
    print(f"{'unit':<15}{'raw_gap':>9}{'hist_gap':>10}{'gap_red%':>9}"
          f"{'raw_twin':>9}{'hist_twin':>10}{'gain%':>8}")
    xg, yg, xt = [], [], []
    for uid in data:
        d = data[uid]
        g = gains.get(uid, np.nan)
        print(f"{uid:<15}{d['raw']['mean_neighbor_soh_gap']:>9.4f}"
              f"{d['history']['mean_neighbor_soh_gap']:>10.4f}"
              f"{d['gap_reduction_pct']:>+9.1f}"
              f"{d['raw']['twin_rate']:>9.3f}{d['history']['twin_rate']:>10.3f}"
              f"{g:>+8.1f}")
        if np.isfinite(g):
            xg.append(d["gap_reduction_pct"])
            xt.append(d["twin_reduction_pct"])
            yg.append(g)

    print("\n--- correlation: more gap reduction -> larger temporal gain? ---")
    for name, xs in [("gap_reduction_pct", xg), ("twin_reduction_pct", xt)]:
        rho = spearmanr(xs, yg)
        print(f"  {name:<20} vs gain: rho={rho.statistic:+.3f} p={rho.pvalue:.3f}")

    n_down = sum(1 for d in data.values() if d["gap_reduction_pct"] < 0)
    print(f"\nverdict: History reduces neighbor-SOH-gap in {n_down}/{len(data)} units")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
