"""bexp14: the GBDT baseline -- the missing standard baseline, and a test of the
mechanism behind pathway 1.

Motivation, twofold:
  1. the battery experiments had only run TabPFN and an MLP, never a tree model.
     GBDT is the standard baseline for tabular tasks and a reviewer will ask for
     it.
  2. the mechanism behind pathway 1 is that a row-invariant model cannot obtain
     cross-row information by itself and must have it materialised. GBDT is also
     row-invariant, so if the account holds, GBDT should benefit from the history
     stack in the same direction as TabPFN, while the harm to the MLP should be
     attributed to overfitting in high dimensions rather than to any ability to
     learn the transform.

================== frozen predictions (committed before the run) ==================
G1: the history stack helps GBDT (delta%<0) in >= 7 of 11 units (same direction as
    the 8/11 for TabPFN)
G2: if GBDT is harmed, the worst harm is < +100% (against +173.7% to +534.4% for
    the MLP)

Criteria:
  G1 holds     -> the "row-invariant, so it must be materialised" account is
                  supported
  G1 fails(<=5)-> the account has a hole and must be rewritten as "specific to
                  TabPFN", recorded as measured
  6/11         -> borderline, no directional claim
==========================================================================

Protocol: reference protocol (normalize="author_minmax"), identical to bexp1.
Scale: 11 units x {raw, history} x 5 seeds.
Also run with context_cap = full (GBDT has no context limit, which is its natural
setting); the cap is recorded so the two readings -- "same budget as TabPFN" and
"GBDT's natural setting" -- can be told apart.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import CONTEXT_CAP, evaluate_unit

OUT = Path("results/battery/bexp14_gbdt.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CAPS = [CONTEXT_CAP, 10 ** 9]      # 2048 (same budget as TabPFN) / full (GBDT natural)

# MLP reference (from bexp1, cap=2048)
MLP_RANGE = (173.7, 534.4)


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["view"], r["seed"], r["cap"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--caps", nargs="+", type=int, default=CAPS)
    args = ap.parse_args()

    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        ds, bk = umap[uid]
        unit = load_fit_unit(ds, bk)
        for cap in args.caps:
            for view in ("raw", "history"):
                for seed in args.seeds:
                    if (uid, view, seed, cap) in done:
                        continue
                    t0 = time.time()
                    r = evaluate_unit(unit, view, seed, backbone="gbdt",
                                      normalize="author_minmax",
                                      context_cap=cap)
                    r["cap"] = cap
                    with OUT.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"{uid:<15}cap={'FULL' if cap>=10**9 else cap:<5}"
                          f"{view:<8}seed={seed} "
                          f"overall={r['rmse_overall']:.6f} "
                          f"macro={r['rmse_cell_macro']:.6f} "
                          f"[{time.time()-t0:.1f}s]", flush=True)
    report()


def _mean(rows, uid, view, cap, key="rmse_overall"):
    v = [r[key] for r in rows if r["unit"] == uid and r["view"] == view
         and r["cap"] == cap]
    return float(np.mean(v)) if v else np.nan


def report():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    b1 = Path("results/battery/bexp1_author_protocol.jsonl")
    ref = [json.loads(l) for l in b1.read_text().splitlines()] if b1.exists() else []

    for cap in sorted({r["cap"] for r in rows}):
        tag = "FULL" if cap >= 10 ** 9 else str(cap)
        print(f"\n{'='*78}\nGBDT history effect "
              f"(cap={tag}, overall RMSE, mean of 5 seeds)")
        print(f"{'unit':<16}{'raw':>10}{'history':>10}{'delta%':>9}"
              f"{'TabPFN Δ%':>11}{'MLP Δ%':>10}")
        deltas, tab_d, mlp_d = [], [], []
        for uid, _, _ in FIT_UNITS:
            rw = _mean(rows, uid, "raw", cap)
            hs = _mean(rows, uid, "history", cap)
            if not np.isfinite(rw) or not np.isfinite(hs):
                continue
            d = (hs - rw) / rw * 100
            deltas.append((uid, d))
            # bexp1 reference
            def rd(bb):
                a = [r["rmse_overall"] for r in ref if r["unit"] == uid
                     and r["view"] == "raw" and r["backbone"] == bb]
                b = [r["rmse_overall"] for r in ref if r["unit"] == uid
                     and r["view"] == "history" and r["backbone"] == bb]
                return (np.mean(b) - np.mean(a)) / np.mean(a) * 100 if a and b else np.nan
            t, m = rd("tabpfn"), rd("mlp")
            tab_d.append(t); mlp_d.append(m)
            print(f"{uid:<16}{rw:>10.6f}{hs:>10.6f}{d:>+8.1f}"
                  f"{t:>+11.1f}{m:>+10.1f}")

        n_good = sum(1 for _, d in deltas if d < 0)
        worst = max(d for _, d in deltas) if deltas else np.nan
        print(f"\n  history helps: {n_good}/{len(deltas)} units  "
              f"worst harm: {worst:+.1f}%")
        print(f"  reference TabPFN helps: {sum(1 for x in tab_d if x<0)}/{len(tab_d)}  "
              f"MLP helps: {sum(1 for x in mlp_d if x<0)}/{len(mlp_d)} "
              f"(MLP harm range +{MLP_RANGE[0]}% ~ +{MLP_RANGE[1]}%)")
        if cap == CONTEXT_CAP:
            g1 = n_good >= 7
            g2 = worst < 100
            print(f"\n  === frozen verdict (cap=2048, same budget as TabPFN) ===")
            print(f"  G1 (helps >= 7/11): {n_good}/{len(deltas)} -> "
                  + ("holds" if g1 else ("borderline" if n_good==6 else "falsified")))
            print(f"  G2 (worst harm < +100%): {worst:+.1f}% -> "
                  + ("holds" if g2 else "falsified"))
            print("  pathway-1 'row-invariant -> must materialise' account: "
                  + ("supported" if g1 else ("uncertain" if n_good==6 else "has a hole")))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
