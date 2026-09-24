"""bexp56: strict-online Pipeline B re-run (W01) -- a thin driver reusing bexp50's assembly
and model registry.

Background (W01, verified):
  The old reading path took mean+/-3sigma over each cell's **full lifetime including capacity**
  and deleted rows per cell -> the row set and the anchor depended on the target's future and
  on the label; and the "first cycle" was in fact the first surviving row after rows 1-4
  (quantified, see the execution ledger).

This script does one thing: reload the 11 units with `reader="strict_online"` and re-run the
same matrix and models under the **deployable pipeline** (normalize="source"; source-fitted
statistics + fixed-scale cycle column), writing results to new paths while leaving the old
archives untouched. Matrix assembly and model implementations **reuse bexp50 directly**
(including its raw/srl/history views and the gbdt/ridge implementations), avoiding protocol
drift from a second harness.

Scale: 11 units x 5 seeds (0-4) x {gbdt: raw,srl,history | ridge: raw,srl} = 275 cells (CPU).
      The 165 TabPFN cells (raw,srl,history) run on a GPU machine via `bexp57_strict_tabpfn.py`
      and are written to the same JSONL (same file name and fields); summarization groups by
      backbone.

================== frozen criteria (frozen after the commit of this script and of bexp50's history branch, before the run) ==================
C0 completeness: all 275 cells have a result or an explicit failure; every cell carries
   reader/rows_kept/first_record_index/seed.
C1 anchor legality: first_record_index is 0 for every loaded unit, and rows_kept >= the
   author-protocol row count.
B1 deployment benefit (GBDT): units improved for history vs raw >= 7/11 -> keep the statement
   with the new values; otherwise downgrade/scope it.
B3 compact SRL: units improved for srl vs raw (if below 7/11 the text must state that the full
   stack is needed).
Note: this script does not decompose the old-vs-new row-set difference into "prediction change
      vs sample-set change" -- that decomposition needs a recomputation on the shared-record
      intersection and is a pending item in the execution ledger; here we report the row counts
      and the first anchor.
==================================================================================

Run:  ./.venv/bin/python experiments/bexp56_strict_rerun.py            # CPU, 275 cells
      ./.venv/bin/python experiments/bexp56_strict_rerun.py --report
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import CONTEXT_CAP

OUT = ROOT / "results/battery/bexp56_strict_rerun.jsonl"
SUM = ROOT / "results/battery/bexp56_strict_rerun_summary.json"
CAP = CONTEXT_CAP
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = {"gbdt": ("raw", "srl", "history"), "ridge": ("raw", "srl")}
EXPECTED_CPU = len(FIT_UNITS) * len(SEEDS) * 5  # gbdt 3 + ridge 2
_units = {}  # unit cache: avoid re-reading from disk for every cell


def _load_bexp50():
    spec = importlib.util.spec_from_file_location(
        "bexp50", ROOT / "experiments" / "bexp50_linear_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B50 = _load_bexp50()
assert {"gbdt", "ridge"} <= set(B50.MODELS), f"bexp50 model registry: {sorted(B50.MODELS)}"


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["backbone"], r["view"], r["seed"]))
    return keys


def run_one(uid, dataset, batch_key, view, backbone, seed):
    unit = _units.setdefault(
        uid, load_fit_unit(dataset, batch_key, reader="strict_online"))
    first_idx = {c.cell_id: int(c.record_index[0]) for c in unit.cells}
    rows_kept = {c.cell_id: int(len(c.y)) for c in unit.cells}
    assert all(v == 0 for v in first_idx.values()), \
        f"{uid}: first record_index != 0 -- suspicious reader"
    t0 = time.time()
    r = B50.evaluate(unit, view, backbone, seed, "source", CAP)
    r.update({"reader": "strict_online", "rows_kept": rows_kept,
              "first_record_index": first_idx, "unit": uid,
              "backbone": backbone})  # bexp50 uses the "model" key; unified to backbone here
    r["runtime_s"] = time.time() - t0
    return r


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()] if OUT.exists() else []
    if not rows:
        print("no results")
        return None
    units = [u for u, _, _ in FIT_UNITS]
    agg = {}
    for r in rows:
        agg.setdefault((r["unit"], r["backbone"], r["view"]), []).append(
            r["rmse_cell_macro"])
    out = {"_meta": {"n_cells": len(rows),
                     "seeds": sorted({r["seed"] for r in rows}),
                     "backbones": sorted({r["backbone"] for r in rows}),
                     "reader": "strict_online", "cap": CAP,
                     "expected_cpu_cells": EXPECTED_CPU,
                     "cpu_complete": len(rows) >= EXPECTED_CPU},
           "per_unit": {}, "verdicts": {}}
    for u in units:
        out["per_unit"].setdefault(u, {})
        for (uu, bk, v), vals in agg.items():
            if uu == u:
                out["per_unit"][u][f"{bk}|{v}"] = {
                    "median": float(np.median(vals)), "n_seeds": len(vals)}
    for bk in sorted({r["backbone"] for r in rows}):
        for v in ("srl", "history"):
            gaps = []
            for u in units:
                a = out["per_unit"][u].get(f"{bk}|{v}")
                b = out["per_unit"][u].get(f"{bk}|raw")
                if a and b:
                    gaps.append((a["median"] - b["median"]) / b["median"] * 100)
            if gaps:
                out["verdicts"][f"{bk}|{v}_vs_raw"] = {
                    "median_delta_pct": float(np.median(gaps)),
                    "units_improved": int(sum(g < 0 for g in gaps)),
                    "n_units": len(gaps)}
    json.dump(out, SUM.open("w"), indent=1)
    return out


def report():
    if not SUM.exists():
        print("no summary")
        return
    out = json.load(SUM.open())
    m = out["_meta"]
    print(f"strict-online Pipeline B: {m['n_cells']} cells | reader={m['reader']} | cap={CAP} | "
          f"CPU complete={m['cpu_complete']} (expected {m['expected_cpu_cells']})")
    for u, d in out["per_unit"].items():
        for k, v in sorted(d.items()):
            bk, view = k.split("|")
            raw = d.get(f"{bk}|raw", {}).get("median")
            dl = ((v["median"] - raw) / raw * 100) if (raw and view != "raw") else None
            print(f"  {u:<16}{k:<18}{v['median']:>10.5f}"
                  f"{(f'{dl:+.1f}%' if dl is not None else '-'):>10}")
    print("\n--- criteria B1 / B3 ---")
    for k, v in out["verdicts"].items():
        ok = v["units_improved"] >= 7
        print(f"  {k:<24} median {v['median_delta_pct']:+.1f}% | "
              f"improved {v['units_improved']}/{v['n_units']} -> "
              f"{'keep' if ok else 'downgrade/scope'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["gbdt", "ridge"])
    args = ap.parse_args()
    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in umap:
        ds, bk = umap[uid]
        for backbone in args.backends:
            for view in CONFIGS[backbone]:
                for seed in SEEDS:
                    if (uid, backbone, view, seed) in done:
                        continue
                    rec = run_one(uid, ds, bk, view, backbone, seed)
                    with OUT.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{uid:<16}{backbone:<7}{view:<8}seed={seed} "
                          f"macro={rec['rmse_cell_macro']:.5f} "
                          f"rows={sum(rec['rows_kept'].values())} "
                          f"[{rec['runtime_s']:.1f}s]", flush=True)
    summarize()
    report()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
