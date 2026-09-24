"""bexp57: the TabPFN arm of strict-online Pipeline B (GPU) -- the GPU part of the W01 matrix.

Relation to bexp56:
  bexp56 runs the CPU part (gbdt/ridge, 275 cells) and writes the summary;
  this script runs the GPU part (tabpfn, 11 units x 5 seeds x {raw, srl, history} = 165 cells)
  into the **same JSONL** (fields identical to bexp56, including backbone/reader/rows_kept/
  first_record_index), writing as it goes and resuming idempotently. Together they are 440 cells.

Device note:
  The paper's TabPFN results were produced pinned to CPU by `residual_lab.models._TABPFN_CPU_KW`
  (the paper states "All runs are executed on CPU"). A single CPU predict measured 42.5 s
  (2048x81 context, 600 test rows), which is infeasible for this matrix; this arm therefore uses
  **CUDA** (measured 0.29 s, 146x faster), changing only the device -- the remaining
  hyper-parameters such as n_estimators match the CPU version -- and writes the device into
  every record for the summary and for the paper's disclosure as measured.

================== frozen criteria (frozen after this script's commit, before the run) ==================
C0 completeness: all 165 cells have a result or an explicit failure; every cell carries
   device=cuda and rows_kept/first_record_index.
C1 anchor legality: first_record_index is 0 for every unit.
B2 TabPFN deployment benefit: units improved for history vs raw >= 7/11 -> keep the statement;
   otherwise scope it / downgrade.
(Combined with bexp56's B1/B3 for the verdict written into the paper; when both backbones are
 <7/11, the deployability claim is downgraded overall to "features are computable" rather than
 "validated benefit".)
==================================================================

Run (on a GPU machine):
  TABPFN_WEIGHTS_DIR=$HOME/.cache/tabpfn TABPFN_DEVICE=cuda \
    ~/.venv-test/bin/python experiments/bexp57_strict_tabpfn.py
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

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import CONTEXT_CAP

OUT = ROOT / "results/battery/bexp56_strict_rerun.jsonl"   # shared with bexp56
CAP = CONTEXT_CAP
SEEDS = [0, 1, 2, 3, 4]
VIEWS = ("raw", "srl", "history")


def _load_bexp50():
    spec = importlib.util.spec_from_file_location(
        "bexp50", ROOT / "experiments" / "bexp50_linear_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B50 = _load_bexp50()
_units = {}


def fit_predict_tabpfn_cuda(X_tr, y_tr, X_te, seed):
    """TabPFN regressor with the same hyper-parameters as the CPU version; only the device is overridable."""
    import torch
    from tabpfn import TabPFNRegressor
    from residual_lab.models import _TABPFN_REG_CKPT, _tabpfn_model_path

    device = os.environ.get("TABPFN_DEVICE",
                            "cuda" if torch.cuda.is_available() else "cpu")
    m = TabPFNRegressor(random_state=seed,
                        model_path=_tabpfn_model_path(_TABPFN_REG_CKPT),
                        n_estimators=2, device=device,
                        ignore_pretraining_limits=True)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


B50.MODELS["tabpfn"] = fit_predict_tabpfn_cuda


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r.get("backbone"), r["view"], r["seed"]))
    return keys


def run_one(uid, dataset, batch_key, view, seed):
    unit = _units.setdefault(
        uid, load_fit_unit(dataset, batch_key, reader="strict_online"))
    first_idx = {c.cell_id: int(c.record_index[0]) for c in unit.cells}
    rows_kept = {c.cell_id: int(len(c.y)) for c in unit.cells}
    assert all(v == 0 for v in first_idx.values()), \
        f"{uid}: first record_index != 0 -- suspicious reader"
    t0 = time.time()
    r = B50.evaluate(unit, view, "tabpfn", seed, "source", CAP)
    r.update({"reader": "strict_online", "rows_kept": rows_kept,
              "first_record_index": first_idx, "unit": uid,
              "backbone": "tabpfn",
              "device": os.environ.get("TABPFN_DEVICE", "auto")})
    r["runtime_s"] = time.time() - t0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        ds, bk = umap[uid]
        for view in VIEWS:
            for seed in args.seeds:
                if (uid, "tabpfn", view, seed) in done:
                    continue
                rec = run_one(uid, ds, bk, view, seed)
                with OUT.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{uid:<16}tabpfn {view:<8}seed={seed} "
                      f"macro={rec['rmse_cell_macro']:.5f} "
                      f"rows={sum(rec['rows_kept'].values())} "
                      f"[{rec['runtime_s']:.1f}s]", flush=True)
    n = sum(1 for _ in OUT.read_text().splitlines()) if OUT.exists() else 0
    print(f"\n[bexp57] batch finished; JSONL now has {n} rows (CPU 275 + TabPFN 165 = 440 for the full matrix)")


if __name__ == "__main__":
    main()
