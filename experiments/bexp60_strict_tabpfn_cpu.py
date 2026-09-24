"""bexp60: the TabPFN arm of strict-online Pipeline B (CPU, local v3 weights) -- the last block
of the W01 matrix.

Background (W01):
  The old reading path took mean+/-3sigma over each cell's **full lifetime including capacity**
  and deleted rows per cell -> the row set and the anchor depended on the target's future and on
  the label. bexp56 has re-run the CPU 275 cells (gbdt/ridge) under the corrected reader
  (reader="strict_online"), and criteria B1/B3 failed (+4.3%/5/11 and +0.3%/5/11). The 165 TabPFN
  cells were to run on a remote GPU via `bexp57_strict_tabpfn.py`, but that host was unreachable
  this round; meanwhile this machine can obtain the same v3 weights as the paper
  (Prior-Labs/tabpfn_3's tabpfn-v3-regressor-v3_default.ckpt, placed in external/tabpfn_weights/)
  and a single-cell predict measures in seconds, so this script re-runs them on **local CPU**.
  The only difference is the device; bexp57 is kept as the GPU variant, and every record here
  explicitly carries device="cpu" and the weights path used.

Configuration (field-for-field identical to bexp56 except backbone/device):
  - reader="strict_online"; normalize="source" (source-fitted statistics + fixed-scale cycle)
  - cap=2048; seeds 0..4; views raw/srl/history; 11 units; backbone="tabpfn" (frozen model)
Output: results/battery/bexp60_strict_tabpfn_cpu.jsonl (per cell) and _summary.json (summary).

================== frozen criteria (frozen after this script's commit, before the run) ==================
C0 completeness: all 165 cells have a result or an explicit failure; every cell carries device,
           weights, rows_kept, first_record_index, seed.
C1 anchor legality: first_record_index is 0 for every loaded unit.
B2 TabPFN deployment benefit: units improved for history vs raw >= 7/11 -> keep the statement;
           otherwise combined with bexp56's B1/B3: when both backbones are <7/11 the deployability
           claim is downgraded overall to "features are computable" rather than "validated benefit".
Note: no accuracy threshold is judged; results are reported as measured whether low or high
(including failed cells).
======================================================================

Run:  ./.venv/bin/python experiments/bexp60_strict_tabpfn_cpu.py            # 165 cells
      ./.venv/bin/python experiments/bexp60_strict_tabpfn_cpu.py --report
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Same v3 weights directory as the paper; when unset, points at the weights directory outside the repo.
os.environ.setdefault("TABPFN_WEIGHTS_DIR",
                      str(ROOT / "external" / "tabpfn_weights"))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import CONTEXT_CAP, fit_predict_tabpfn

OUT = ROOT / "results/battery/bexp60_strict_tabpfn_cpu.jsonl"
SUM = ROOT / "results/battery/bexp60_strict_tabpfn_cpu_summary.json"
CAP = CONTEXT_CAP
SEEDS = [0, 1, 2, 3, 4]
VIEWS = ("raw", "srl", "history")
BACKBONE = "tabpfn"
EXPECTED = len(FIT_UNITS) * len(SEEDS) * len(VIEWS)  # 165
_units = {}


def _load_bexp50():
    spec = importlib.util.spec_from_file_location(
        "bexp50", ROOT / "experiments" / "bexp50_linear_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B50 = _load_bexp50()


def _tabpfn(X_tr, y_tr, X_te, seed):
    return fit_predict_tabpfn(X_tr, y_tr, X_te, seed)


# Reuse bexp50's matrix assembly and scoring; add only one model entry (avoiding protocol drift
# from a second harness).
B50.MODELS[BACKBONE] = _tabpfn


def _weights_path():
    from residual_lab.models import _TABPFN_REG_CKPT
    return str(_TABPFN_REG_CKPT)


def run_one(uid, dataset, batch_key, view, seed):
    unit = _units.setdefault(
        uid, load_fit_unit(dataset, batch_key, reader="strict_online"))
    first_idx = {c.cell_id: int(c.record_index[0]) for c in unit.cells}
    rows_kept = {c.cell_id: int(len(c.y)) for c in unit.cells}
    assert all(v == 0 for v in first_idx.values()), \
        f"{uid}: first record_index != 0 -- suspicious reader"
    t0 = time.time()
    r = B50.evaluate(unit, view, BACKBONE, seed, "source", CAP)
    r.update({"reader": "strict_online", "rows_kept": rows_kept,
              "first_record_index": first_idx, "unit": uid,
              "backbone": BACKBONE, "device": "cpu",
              "weights": _weights_path()})
    r["runtime_s"] = time.time() - t0
    return r


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["view"], r["seed"]))
    return keys


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()] if OUT.exists() else []
    if not rows:
        print("no results")
        return None
    units = [u for u, _, _ in FIT_UNITS]
    agg = {}
    for r in rows:
        agg.setdefault((r["unit"], r["view"]), []).append(r["rmse_cell_macro"])
    out = {"_meta": {"n_cells": len(rows), "expected": EXPECTED,
                     "seeds": sorted({r["seed"] for r in rows}),
                     "views": sorted({r["view"] for r in rows}),
                     "backbone": BACKBONE,
                     "device": sorted({r["device"] for r in rows}),
                     "weights": sorted({r["weights"] for r in rows}),
                     "reader": "strict_online", "cap": CAP,
                     "complete": len(rows) >= EXPECTED},
           "per_unit": {}, "verdicts": {}}
    for u in units:
        out["per_unit"].setdefault(u, {})
        for (uu, v), vals in agg.items():
            if uu == u:
                out["per_unit"][u][v] = {"median": float(np.median(vals)),
                                         "n_seeds": len(vals)}
    for v in ("srl", "history"):
        gaps = []
        for u in units:
            a = out["per_unit"][u].get(v)
            b = out["per_unit"][u].get("raw")
            if a and b:
                gaps.append((a["median"] - b["median"]) / b["median"] * 100)
        if gaps:
            out["verdicts"][f"{v}_vs_raw"] = {
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
    print(f"strict-online Pipeline B | TabPFN: {m['n_cells']}/{m['expected']} cells | "
          f"device={m['device']} | reader={m['reader']} | cap={CAP}")
    for u, d in out["per_unit"].items():
        for v in VIEWS:
            if v not in d:
                continue
            raw = d.get("raw", {}).get("median")
            dl = ((d[v]["median"] - raw) / raw * 100) if (raw and v != "raw") else None
            print(f"  {u:<16}{BACKBONE}|{v:<8}{d[v]['median']:>10.5f}"
                  f"{(f'{dl:+.1f}%' if dl is not None else '-'):>10}")
    print("\n--- criteria B2 ---")
    for k, v in out["verdicts"].items():
        ok = v["units_improved"] >= 7
        print(f"  {k:<16} median {v['median_delta_pct']:+.1f}% | "
              f"improved {v['units_improved']}/{v['n_units']} -> "
              f"{'keep' if ok else 'downgrade/scope'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--views", nargs="+", default=list(VIEWS))
    args = ap.parse_args()
    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        ds, bk = umap[uid]
        for view in args.views:
            for seed in SEEDS:
                if (uid, view, seed) in done:
                    continue
                rec = run_one(uid, ds, bk, view, seed)
                with OUT.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{uid:<16}{BACKBONE:<8}{view:<8}seed={seed} "
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
