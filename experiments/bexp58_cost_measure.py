"""bexp58: the **measured** convention for computational cost (W06 measurement part).

Background:
  Section 3.6 previously reported "~3 s/unit" and fleet-level CPU-hours/energy; the audit found
  three problems: (a) the denominator counted training rows as prediction rows; (b) wall-clock
  was treated as CPU time; (c) batch and single-row costs were not measured separately and no
  memory/thread records were kept. This script re-measures per revision_execution_plan section W06.

Scenario (matching the measured object):
  - Fitted model + prepared context; the target cell delivers **one** computable current-cycle
    feature row per arrival;
  - feature state updates incrementally (this script measures only the model-side cost, not
    feature updates).

Measured quantities (per (unit, view) combination):
  1. fit_s      : one fit on the training matrix (GBDT, library default hyper-parameters, seed 0)
  2. lat1_ms    : **single-row** prediction latency (batch=1), >=20 repeats, median/p95 reported
  3. thr_us_row : **fixed-batch** throughput (batch=1024 rows), in microseconds per row
  4. rss_mb     : peak process RSS delta during fit+predict (via /proc or psutil if available,
                  otherwise reported as unmeasurable)
  Records: CPU model/core count, thread count, python/numpy/sklearn versions, start time.

================== frozen criteria (frozen after this script's commit, before the run) ==================
C0 completeness: every (unit, view, item) has a value or an explicitly recorded failure reason;
                latency/throughput repeats n >= 20 (if fewer, state the reason and the actual
                count), with median and p95 reported.
C1 convention: every per-cell/per-row cost uses the **actual number of prediction rows** as the
               denominator; "CPU-hours" must not appear (unless process CPU time is also
               reported; this script does not measure CPU time).
C2 records: thread count, versions, hardware, start time are written to the output; anything
               unmeasurable (e.g. RSS) is written as "unmeasurable" -- a guess must not be
               substituted.
The criteria set no "must be below X" pass line: results are reported as measured whether high
or low.
==================================================================

Run:  ./.venv/bin/python experiments/bexp58_cost_measure.py            # full run
      ./.venv/bin/python experiments/bexp58_cost_measure.py --report
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import CONTEXT_CAP, subsample_context

OUT = ROOT / "results/battery/bexp58_cost_measure.json"
UNITS = ["XJTU-2C", "MIT"]          # one small training set / one large training set
VIEWS = ["raw", "srl"]
REPEATS = 24                        # >=20
BATCH = 1024


def _rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # macOS: bytes -> MB
    except Exception:
        return None


def _env():
    import sklearn
    import numpy
    try:
        from threadpoolctl import threadpool_info
        thr = threadpool_info()
        nthreads = sorted({t.get("num_threads") for t in thr}) if thr else None
    except Exception:
        nthreads = None
    return {
        "python": sys.version.split()[0], "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "threads": nthreads,
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _build_xy(Z, view, cycle_mode):
    sys.path.insert(0, str(ROOT / "experiments"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b50", ROOT / "experiments" / "bexp50_linear_baseline.py")
    b50 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b50)
    return b50.build_xy(Z, view, cycle_mode)


def measure_one(uid, dataset, batch_key, view):
    import importlib.util
    from battery_lab.data_adapters import normalize_cells
    from sklearn.ensemble import HistGradientBoostingRegressor

    unit = load_fit_unit(dataset, batch_key)     # author-protocol features (this measurement cares only about cost)
    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "source", source_cells=tr)
    cm = "scaled200"
    Xtr = np.vstack([_build_xy(Xn[c.cell_id], view, cm) for c in tr])
    ytr = np.concatenate([c.y for c in tr])
    Xte = np.vstack([_build_xy(Xn[c.cell_id], view, cm) for c in te])
    Xc, yc = subsample_context(Xtr, ytr, CONTEXT_CAP, 0)

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    m = HistGradientBoostingRegressor(random_state=0).fit(Xc, yc)
    fit_s = time.perf_counter() - t0

    # single-row latency (batch=1)
    lat = []
    for _ in range(REPEATS):
        x = Xte[np.random.default_rng(_).integers(len(Xte))][None, :]
        t = time.perf_counter(); m.predict(x); lat.append((time.perf_counter() - t) * 1e3)
    # fixed-batch throughput
    thr = []
    for r in range(max(3, REPEATS // 4)):
        xb = Xte[np.random.default_rng(100 + r).integers(0, len(Xte), BATCH)]
        t = time.perf_counter(); m.predict(xb); thr.append((time.perf_counter() - t) / BATCH * 1e6)
    rss1 = _rss_mb()
    return {
        "unit": uid, "view": view, "dim": int(Xc.shape[1]),
        "n_ctx_rows": int(len(yc)), "n_test_rows": int(len(Xte)),
        "fit_s": fit_s,
        "lat1_ms": {"median": statistics.median(lat),
                    "p95": sorted(lat)[max(0, int(0.95 * len(lat)) - 1)],
                    "n": len(lat)},
        "thr_us_row": {"median": statistics.median(thr),
                       "p95": sorted(thr)[max(0, int(0.95 * len(thr)) - 1)],
                       "n": len(thr), "batch": BATCH},
        "rss_delta_mb": (rss1 - rss0) if (rss0 and rss1) else "unmeasurable",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=UNITS)
    args = ap.parse_args()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    res = json.load(OUT.open()) if OUT.exists() else {"_env": _env(), "cells": []}
    have = {(c["unit"], c["view"]) for c in res["cells"]}
    for uid in args.units:
        ds, bk = umap[uid]
        for view in VIEWS:
            if (uid, view) in have:
                continue
            rec = measure_one(uid, ds, bk, view)
            res["cells"].append(rec)
            json.dump(res, OUT.open("w"), indent=1)
            print(f"{uid:<10}{view:<5}dim={rec['dim']:<4}fit={rec['fit_s']:.2f}s "
                  f"lat1={rec['lat1_ms']['median']:.2f}ms "
                  f"thr={rec['thr_us_row']['median']:.1f}us/row "
                  f"rss+={rec['rss_delta_mb']}", flush=True)
    print("\n[env]", json.dumps(res["_env"], ensure_ascii=False))


def report():
    res = json.load(OUT.open())
    print("=== bexp58 cost measurement (GBDT, library default hyper-parameters; denominator = actual prediction rows) ===")
    print(f"{'unit':<10}{'view':<6}{'dim':>4}{'fit_s':>8}{'lat1_ms':>10}{'p95':>8}{'us/row':>9}{'n':>4}")
    for c in res["cells"]:
        print(f"{c['unit']:<10}{c['view']:<6}{c['dim']:>4}{c['fit_s']:>8.2f}"
              f"{c['lat1_ms']['median']:>10.2f}{c['lat1_ms']['p95']:>8.2f}"
              f"{c['thr_us_row']['median']:>9.1f}{c['lat1_ms']['n']:>4}")
    print("\n[env]", json.dumps(res["_env"], ensure_ascii=False))
    print("\nnote: single-row latency is batch=1; throughput is batch=%d; process CPU time was not measured, so no CPU-hours are reported."
          % BATCH)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
