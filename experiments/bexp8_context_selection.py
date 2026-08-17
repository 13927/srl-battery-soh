"""bexp8: context-selection strategies, a sweep over k, and an adaptive rule.

Three parts:
  A. strategy x k sweep: {random, knn (k over a grid), cluster} x unit x seed
  B. descriptors: group_shift_auc / knn_locality / degradation_rate per unit
  C. adaptive: choose k with adaptive_k(descriptors) and compare against the best
     fixed k and the random baseline

The context is capped at 8192, leave-one-cell-out is strict, and the history view
is used throughout.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.context_selection import (adaptive_k, distribution_indicators,
                                           retrieval_predict)
from battery_lab.data_adapters import FIT_UNITS, load_fit_unit

OUT = Path("results/battery/bexp8_context_selection.jsonl")
IND = Path("results/battery/bexp8_indicators.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# smaller units first (per-member kNN retrieval is slow on the large ones)
DEFAULT_UNITS = ["XJTU-2C", "XJTU-3C", "XJTU-satellite", "TJU-1", "TJU-3", "MIT"]
K_GRID = [20, 50, 100, 200]
SEEDS = [0, 1, 2]


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add((r["unit"], r["strategy"], r["k_per_test"], r["seed"]))
            except json.JSONDecodeError:
                continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=DEFAULT_UNITS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    done = done_keys()
    unit_map = {u: (d, b) for u, d, b in FIT_UNITS}
    inds = json.load(IND.open()) if IND.exists() else {}

    for uid in args.units:
        ds, bk = unit_map[uid]
        unit = load_fit_unit(ds, bk)
        if uid not in inds:
            inds[uid] = distribution_indicators(unit)
            json.dump(inds, IND.open("w"), indent=1)
            print(f"[descriptors] {uid}: auc={inds[uid]['group_shift_auc']:.3f} "
                  f"knn_r2={inds[uid]['knn_locality_r2']:.3f} "
                  f"deg={inds[uid]['degradation_rate']:.5f} "
                  f"→ adaptive_k={adaptive_k(inds[uid])}", flush=True)

        # task list: random + knn for each k + cluster + the adaptive k
        tasks = [("random", 0), ("cluster", 0)]
        tasks += [("knn", k) for k in K_GRID]
        ak = adaptive_k(inds[uid])
        if ak not in K_GRID:
            tasks.append(("knn", ak))     # the adaptive k, added if not on the grid

        for strat, k in tasks:
            for seed in args.seeds:
                key = (uid, strat, k, seed)
                if key in done:
                    continue
                t0 = time.time()
                r = retrieval_predict(unit, "history", seed, strategy=strat,
                                      k_per_test=k)
                with OUT.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"  {uid:<15}{strat:<8}k={k:<4}seed={seed} "
                      f"macro={r['rmse_cell_macro']:.6f} ({time.time()-t0:.0f}s)",
                      flush=True)
    report()


def report():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    inds = json.load(IND.open()) if IND.exists() else {}
    units = sorted({r["unit"] for r in rows})
    print("\n===== strategy x k sweep (history, cell_macro, mean over seeds) =====")
    for uid in units:
        u = [r for r in rows if r["unit"] == uid]
        rand = np.mean([r["rmse_cell_macro"] for r in u if r["strategy"] == "random"])
        clus = [r["rmse_cell_macro"] for r in u if r["strategy"] == "cluster"]
        line = f"{uid:<15} random={rand:.6f}"
        if clus:
            line += f"  cluster={np.mean(clus):.6f}"
        for k in K_GRID:
            v = [r["rmse_cell_macro"] for r in u if r["strategy"] == "knn"
                 and r["k_per_test"] == k]
            if v:
                line += f"  knn{k}={np.mean(v):.6f}"
        print(line)

    print("\n===== adaptive vs best fixed vs random (improvement over random, "
          "per cent) =====")
    print(f"{'unit':<15}{'auc':>6}{'knn_r2':>8}{'ad. k':>7}"
          f"{'random':>10}{'adaptive':>10}{'best k':>9}{'delta%':>9}")
    for uid in units:
        if uid not in inds:
            continue
        u = [r for r in rows if r["unit"] == uid]
        rand = np.mean([r["rmse_cell_macro"] for r in u if r["strategy"] == "random"])
        ak = adaptive_k(inds[uid])
        av = [r["rmse_cell_macro"] for r in u if r["strategy"] == "knn"
              and r["k_per_test"] == ak]
        knn_by_k = {k: np.mean([r["rmse_cell_macro"] for r in u
                                if r["strategy"] == "knn" and r["k_per_test"] == k])
                    for k in K_GRID
                    if any(r["strategy"] == "knn" and r["k_per_test"] == k for r in u)}
        best_k = min(knn_by_k, key=knn_by_k.get) if knn_by_k else None
        av_v = np.mean(av) if av else (knn_by_k.get(ak) or np.nan)
        d = (av_v - rand) / rand * 100 if np.isfinite(av_v) else np.nan
        print(f"{uid:<15}{inds[uid]['group_shift_auc']:>6.3f}"
              f"{inds[uid]['knn_locality_r2']:>8.3f}{ak:>6}"
              f"{rand:>10.6f}{av_v:>10.6f}"
              f"{('k%d=%.6f'%(best_k,knn_by_k[best_k])) if best_k else '-':>16}"
              f"{d:>+8.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
