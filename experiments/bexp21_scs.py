"""bexp21: source-side configuration selection -- nested leave-one-cell-out
estimates and the selection verdict.

================ frozen criteria (committed before the run) ================
Selection rule (fixed in advance, source_selection.select_config):
  per unit, take the configuration with the smallest source-side estimate; if its
  gap to the runner-up is below the pooled SE, fall back to gbdt-history.
S1 (regret): in >= 9 of 11 units, the regret of the selected config against the
             oracle is < 11.3 per cent (the noise floor)
S2 (main goal): library-level (overall RMSE vs the published PINN) better on 4/4
S3 (safety): no unit selects a config more than 25 per cent worse than the oracle
Failure branches: S2 fails but S1 holds -> demote to "automatic selection with
             bounded regret"; S1 fails -> the GBDT-proxy variant, then close out
=======================================================================

Estimation cost: GBDT on all source cells x 5 seeds (seconds); TabPFN on a
subsample of <=12 source cells (SAMPLING_SEED=20260731) x 3 seeds. Resumable (one
config written at a time). The oracle and the test cells take no part in any
estimate or selection; they are used only to score at the verdict stage.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.source_selection import (CONFIGS, MAX_TABPFN_CELLS,
                                          loco_estimate, select_config)

OUT = Path("results/battery/bexp21_source_estimates.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
NOISE = 25.0 / 25.0  # placeholder (unused); the verdict constants are inlined in report()

PINN_OFFICIAL = {"XJTU": 0.009410, "TJU": 0.015789,
                 "HUST": 0.008685, "MIT": 0.007400}
GRP = {"XJTU": ["XJTU-2C", "XJTU-3C", "XJTU-R2.5", "XJTU-R3", "XJTU-RW",
                "XJTU-satellite"],
       "TJU": ["TJU-1", "TJU-2", "TJU-3"], "HUST": ["HUST"], "MIT": ["MIT"]}


def load_state():
    return json.load(OUT.open()) if OUT.exists() else {}


def save_state(s):
    json.dump(s, OUT.open("w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--backbones", nargs="+", default=["gbdt", "tabpfn"])
    args = ap.parse_args()

    state = load_state()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        unit = None
        state.setdefault(uid, {})
        for bb, view in CONFIGS:
            if bb not in args.backbones:
                continue
            key = f"{bb}-{view}"
            if key in state[uid]:
                continue
            if unit is None:
                unit = load_fit_unit(*umap[uid])
            t0 = time.time()
            if bb == "gbdt":
                est = loco_estimate(unit, bb, view, seeds=[0, 1, 2, 3, 4])
            else:
                est = loco_estimate(unit, bb, view, seeds=[0, 1, 2],
                                    max_cells=MAX_TABPFN_CELLS)
            est["runtime_s"] = time.time() - t0
            state[uid][key] = est
            save_state(state)
            print(f"{uid:<15}{key:<16} est={est['est_macro']:.6f} "
                  f"se={est['se']:.6f} n_cells={est['n_cells']} "
                  f"[{est['runtime_s']:.0f}s]", flush=True)
    report()


def report():
    state = load_state()
    # test-side measurements (for the verdict; from the existing bexp1/bexp14 data,
    # cap=2048, same budget)
    b1 = [json.loads(l) for l in
          Path("results/battery/bexp1_author_protocol.jsonl").read_text().splitlines()]
    b14 = [json.loads(l) for l in
           Path("results/battery/bexp14_gbdt.jsonl").read_text().splitlines()]

    def test_rmse(u, bb, view):
        if bb == "tabpfn":
            x = [r["rmse_overall"] for r in b1 if r["unit"] == u
                 and r["view"] == view and r["backbone"] == "tabpfn"]
        else:
            x = [r["rmse_overall"] for r in b14 if r["unit"] == u
                 and r["view"] == view and r["cap"] == 2048]
        return float(np.mean(x)) if x else np.nan

    NOISE_PCT = 11.3
    print(f"\n{'='*88}\nSCS verdict (selection uses the source side only; "
          f"scoring uses the official test cells)")
    print(f"{'unit':<15}{'selected':<16}{'reason':<12}{'measured':>10}{'oracle':>9}"
          f"{'regret%':>9}")
    sel_rmse, s1_ok, s3_bad = {}, 0, 0
    n_done = 0
    for uid in [u for u, _, _ in FIT_UNITS]:
        if uid not in state or len(state[uid]) < 4:
            print(f"{uid:<15} (estimates incomplete: {len(state.get(uid, {}))}/4)")
            continue
        n_done += 1
        sel = select_config(state[uid])
        bb, view = sel["selected"].split("-")
        got = test_rmse(uid, bb, view)
        orc = min(test_rmse(uid, b, v) for b, v in
                  [("tabpfn", "raw"), ("tabpfn", "history"),
                   ("gbdt", "raw"), ("gbdt", "history")])
        regret = (got - orc) / orc * 100
        sel_rmse[uid] = got
        if regret < NOISE_PCT:
            s1_ok += 1
        if regret > 25:
            s3_bad += 1
        print(f"{uid:<15}{sel['selected']:<16}{sel['reason']:<12}"
              f"{got:>9.5f}{orc:>9.5f}{regret:>+9.2f}")

    if n_done < 11:
        print(f"\n({n_done}/11 units done, verdict pending the rest)")
        return
    print(f"\nlibrary level vs the published PINN (overall RMSE):")
    s2_ok = 0
    for lib, us in GRP.items():
        v = float(np.mean([sel_rmse[u] for u in us]))
        d = (v - PINN_OFFICIAL[lib]) / PINN_OFFICIAL[lib] * 100
        if v < PINN_OFFICIAL[lib]:
            s2_ok += 1
        print(f"  {lib:<6} SCS={v:.6f} vs published={PINN_OFFICIAL[lib]:.6f} -> {d:+.1f}%")
    print(f"\n=== frozen verdict ===")
    print(f"  S1 (regret<{NOISE_PCT}% in >=9/11): {s1_ok}/11 -> "
          + ("holds" if s1_ok >= 9 else "falsified"))
    print(f"  S2 (library level better than published on 4/4): {s2_ok}/4 -> "
          + ("holds" if s2_ok == 4 else "falsified"))
    print(f"  S3 (no catastrophic pick >25%): {s3_bad} -> "
          + ("holds" if s3_bad == 0 else "falsified"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
