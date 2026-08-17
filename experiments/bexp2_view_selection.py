"""bexp2: nested leave-one-cell-out view selection inside the source cells, plus a
retrospective check.

The source-side results are cached per (unit, seed) so a run can resume; the
summary applies the majority vote and compares it with the test-side winner from
bexp1 (the retrospective check).

Legality: only official training cells are ever used; official test cells take no
part.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.view_selection import decide_from_per_cell, source_internal_loco

OUT = Path("results/battery/bexp2_view_selection.jsonl")
BEXP1 = Path("results/battery/bexp1_author_protocol.jsonl")
SEEDS = [0, 1, 2, 3, 4]


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add((r["unit"], r["seed"]))
            except json.JSONDecodeError:
                continue
    return keys


def main():
    done = done_keys()
    for uid, ds, bk in FIT_UNITS:
        unit = load_fit_unit(ds, bk)
        for seed in SEEDS:
            if (uid, seed) in done:
                continue
            pc = source_internal_loco(unit, seed, backbone="tabpfn")
            d = decide_from_per_cell(pc, boot_seed=seed)
            rec = {"unit": uid, "seed": seed, "decision": d["decision"],
                   "mean_diff": d["mean_diff"], "ci": d["ci"],
                   "n_cells": d["n_cells"], "per_cell": pc}
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{uid:<14} seed={seed} decision={d['decision']:<8} "
                  f"diff={d['mean_diff']:+.6f} CI={d['ci']}", flush=True)
    summarize()


def test_side_winner():
    """Take the test-side winner per unit for TabPFN from bexp1 (cell_macro, mean
    of 5 seeds)."""
    rows = [json.loads(l) for l in BEXP1.read_text().splitlines()]
    winners = {}
    for uid, _, _ in FIT_UNITS:
        m = {}
        for view in ("raw", "history"):
            vals = [r["rmse_cell_macro"] for r in rows
                    if r["unit"] == uid and r["view"] == view
                    and r["backbone"] == "tabpfn"]
            if vals:
                m[view] = float(np.mean(vals))
        if len(m) == 2:
            winners[uid] = ("history" if m["history"] < m["raw"] else "raw",
                            m["raw"], m["history"])
    return winners


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    winners = test_side_winner()
    print("\n===== retrospective check: source-side rule vs test-side winner "
          "=====")
    hits, total = 0, 0
    print(f"{'unit':<14} {'votes':<22} {'decision':<10} {'test winner':<12} "
          f"{'raw macro':>10} {'hist macro':>11} hit")
    for uid, _, _ in FIT_UNITS:
        votes = [r["decision"] for r in rows if r["unit"] == uid]
        if len(votes) < len(SEEDS) or uid not in winners:
            continue
        counts = {v: votes.count(v) for v in ("raw", "history", "abstain")}
        top = max(counts.values())
        cand = [v for v, c in counts.items() if c == top]
        final = cand[0] if len(cand) == 1 else "abstain"
        w, raw_m, hist_m = winners[uid]
        hit = (final == w) or (final == "abstain")  # abstain is not counted wrong
                                                   # but is flagged separately
        mark = "HIT" if final == w else ("ABST" if final == "abstain" else "MISS")
        if final == w:
            hits += 1
        total += 1
        print(f"{uid:<14} {str(votes):<22} {final:<9} {w:<8} "
              f"{raw_m:9.6f} {hist_m:9.6f} {mark}")
    print(f"\nstrict hits (abstentions excluded): {hits}/{total}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarize()
    else:
        main()
