"""bexp61: second-round measured computational cost (W06 revision / second-round review F06).

Background (the three problems F06 identified, all verified):
  (a) `bexp58_cost_measure.py::_rss_mb` on macOS divided `ru_maxrss` (bytes) only by 1024,
      writing KiB into `rss_delta_mb`; and the difference of process high-water marks is not an
      independent per-configuration peak.
  (b) the batch throughput actually had n=6 repeats (`max(3, REPEATS // 4)`), below the criterion's
      n >= 20, and its p95 value cannot support a tail-latency statement.
  (c) the cost section mixed historical archived timings, local measurements and bexp60 per-cell
      runtimes, and based cross-hardware claims such as "the two backbones differ by three orders
      of magnitude" on that mixture.
This script redoes only (a) and (b) and adds the history view and the feature-construction cost;
(c) is handled by rewriting the paper text.

Measured quantities (per (unit, view) combination; each combination measures peak memory in an
**independent subprocess**):
  1. fit_s      : one fit on the 2048-row context (HistGradientBoostingRegressor, seed 0)
  2. lat1_ms    : single-row prediction latency (batch=1), REPEATS=24 repeats, median/p95 reported
  3. thr_us_row : fixed-batch throughput (batch=1024), THR_REPEATS=24 repeats, median/p95 reported
  4. peak_rss_mib: subprocess peak RSS (macOS: ru_maxrss bytes -> MiB; Linux: Ru_maxrss KiB -> MiB)
  5. feat_ms_per_row: per-cell feature-matrix construction cost / retained rows (>=20 cells
     sampled, median reported)

Scenario (matching the measured object): fitted model + prepared context; the target cell delivers
one computable current-cycle feature row per arrival; feature state updates incrementally. Only the
**model-side + feature-construction** cost is measured, excluding data acquisition.

================== frozen criteria (frozen after this script's commit, before the run) ==================
C0 completeness: every (unit, view) has the five items fit/lat1/thr/peak_rss/feat, or an
   explicitly recorded failure reason.
C1 repeat counts: single-row latency and batch throughput repeats n >= 20; if fewer, state the
   reason and the actual count; both report median and p95 (p95 is the ceil(0.95*n)-1-th after
   sorting; with small n it is not called a "tail-latency confidence").
C2 units: memory is always recorded in MiB with per-platform conversion (macOS bytes /1024/1024;
   Linux KiB /1024); KiB must not be labelled MB.
C3 convention: every per-cell/per-row cost uses the **actual number of prediction rows** as the
   denominator; no CPU-hours, no derived energy; memory is the **subprocess peak** (not the
   difference of process high-water marks).
C4 records: thread count, library versions, hardware, start time are written to the output;
   unmeasurable items are written as "unmeasurable".
The criteria set no pass line: results are reported as measured whether high or low.

Criterion dynamics (v3, 2026-09-23, driven by third-round review F3-07):
  C5 feature-cost convention: `feat_ms_per_row` = the **construction time of a whole cell's feature
     matrix / that cell's retained rows**; it is an **amortized cost**, not the incremental cost of
     one online state update; the paper's wording must state the amortized convention.
  C6 warm-up: discard the **first 2 samples** of the measurement sequence (JIT/cache warm-up). The
     v2 implementation mistakenly used `sorted(per_row)[2:]` (deleting the two smallest after
     sorting); corrected to `per_row[2:]` in **measurement order**.
  (This re-run was only because of C6; the accuracy experiments are unaffected.)

Run:  ./.venv/bin/python experiments/bexp61_cost_measure_v2.py            # full run
      ./.venv/bin/python experiments/bexp61_cost_measure_v2.py --report
      ./.venv/bin/python experiments/bexp61_cost_measure_v2.py --child <unit> <view>   # internal
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import CONTEXT_CAP, subsample_context

OUT = ROOT / "results/battery/bexp61_cost_measure.json"
UNITS = ["XJTU-2C", "MIT"]          # one small training set / one large training set
VIEWS = ["raw", "srl", "history"]
REPEATS = 24                        # >= 20 (single-row latency)
THR_REPEATS = 24                    # >= 20 (batch throughput)
BATCH = 1024
SEED = 0


# --------------------------------------------------------------------------- #
# Environment and units
# --------------------------------------------------------------------------- #
def _env():
    import numpy
    import sklearn
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


def _peak_rss_mib():
    """Peak RSS of this process in MiB (C2). macOS ru_maxrss is bytes, Linux is KiB."""
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return raw / 1024.0 / 1024.0
    return raw / 1024.0


def _p95(values):
    """p95: the ceil(0.95*n)-1-th after sorting (0-based); with small n it is only a descriptive upper bound."""
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1))
    return s[k]


def _build_xy(Z, view, cycle_mode):
    """Reuse bexp50's assembly (consistent raw/srl/history conventions)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b50", ROOT / "experiments" / "bexp50_linear_baseline.py")
    b50 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b50)
    return b50.build_xy(Z, view, cycle_mode)


def _prepare(uid, dataset, batch_key, view):
    """Return (X_ctx, y_ctx, X_te, n_feat_rows, feat_ms_per_row)."""
    from battery_lab.data_adapters import normalize_cells
    unit = load_fit_unit(dataset, batch_key)     # author-protocol features (cost only; same convention as bexp58)
    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "source", source_cells=tr)
    cm = "scaled200"
    Xtr = np.vstack([_build_xy(Xn[c.cell_id], view, cm) for c in tr])
    ytr = np.concatenate([c.y for c in tr])
    Xte = np.vstack([_build_xy(Xn[c.cell_id], view, cm) for c in te])
    Xc, yc = subsample_context(Xtr, ytr, CONTEXT_CAP, SEED)

    # feat_ms_per_row: build the feature matrix per cell, divide by that cell's retained rows; sample >= 20 cells
    cells = (tr + te)
    if len(cells) > 40:
        idx = np.random.default_rng(0).choice(len(cells), 40, replace=False)
        sample = [cells[int(i)] for i in idx]
    else:
        sample = list(cells)
    per_row = []
    for c in sample:
        Z = Xn[c.cell_id]
        t0 = time.perf_counter()
        M = _build_xy(Z, view, cm)
        dt = time.perf_counter() - t0
        per_row.append(dt / max(len(M), 1) * 1e3)
    # discard the first 2 samples of the measurement sequence (JIT/cache warm-up).
    # v3 fix (2026-09-23, F3-07): discard in **measurement order**, not sorted() and removing minima.
    per_row = per_row[2:] if len(per_row) > 3 else per_row
    return Xc, yc, Xte, per_row


def measure_child(uid, dataset, batch_key, view):
    """Single-configuration measurement inside a subprocess (C3: memory is the subprocess peak)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rss0 = _peak_rss_mib()
    Xc, yc, Xte, per_row = _prepare(uid, dataset, batch_key, view)

    t0 = time.perf_counter()
    m = HistGradientBoostingRegressor(random_state=SEED).fit(Xc, yc)
    fit_s = time.perf_counter() - t0

    lat = []
    for i in range(REPEATS):
        x = Xte[np.random.default_rng(i).integers(len(Xte))][None, :]
        t = time.perf_counter()
        m.predict(x)
        lat.append((time.perf_counter() - t) * 1e3)

    thr = []
    for r in range(THR_REPEATS):
        xb = Xte[np.random.default_rng(1000 + r).integers(0, len(Xte), BATCH)]
        t = time.perf_counter()
        m.predict(xb)
        thr.append((time.perf_counter() - t) / BATCH * 1e6)

    peak = _peak_rss_mib()
    return {
        "unit": uid, "view": view, "reader": "author_offline",
        "dim": int(Xc.shape[1]), "n_ctx_rows": int(len(yc)),
        "n_test_rows": int(len(Xte)), "n_feat_cells_sampled": int(len(per_row)),
        "fit_s": fit_s,
        # every record carries its own environment snapshot (C4): avoiding one possibly stale
        # loadavg shared by the whole batch
        "loadavg_at_measure": (list(os.getloadavg())
                               if hasattr(os, "getloadavg") else None),
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lat1_ms": {"median": statistics.median(lat), "p95": _p95(lat), "n": len(lat)},
        "thr_us_row": {"median": statistics.median(thr), "p95": _p95(thr),
                       "n": len(thr), "batch": BATCH},
        "peak_rss_mib": peak, "peak_rss_delta_mib": peak - rss0,
        "feat_ms_per_row": {"median": statistics.median(per_row), "n": len(per_row)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=UNITS)
    ap.add_argument("--views", nargs="+", default=VIEWS)
    ap.add_argument("--child", nargs=2, default=None, metavar=("UNIT", "VIEW"))
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing records and re-measure the whole batch (the old file is renamed and archived by the author separately)")
    args = ap.parse_args()

    umap = {u: (d, b) for u, d, b in FIT_UNITS}

    if args.child:
        uid, view = args.child
        ds, bk = umap[uid]
        rec = measure_child(uid, ds, bk, view)
        print(json.dumps(rec))
        return

    if args.fresh and OUT.exists():
        res = {"_env": _env(), "cells": []}
    else:
        res = json.load(OUT.open()) if OUT.exists() else {"_env": _env(), "cells": []}
    have = {(c["unit"], c["view"]) for c in res["cells"]}
    for uid in args.units:
        for view in args.views:
            if (uid, view) in have:
                continue
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--child", uid, view],
                               capture_output=True, text=True, cwd=str(ROOT))
            if p.returncode != 0:
                print(f"[FAIL] {uid} {view}: {p.stderr.strip().splitlines()[-1:]}", flush=True)
                continue
            rec = json.loads(p.stdout.strip().splitlines()[-1])
            res["cells"].append(rec)
            json.dump(res, OUT.open("w"), indent=1)
            print(f"{uid:<10}{view:<8}dim={rec['dim']:<4}fit={rec['fit_s']:.2f}s "
                  f"lat1={rec['lat1_ms']['median']:.2f}ms(n={rec['lat1_ms']['n']}) "
                  f"thr={rec['thr_us_row']['median']:.1f}us/row(n={rec['thr_us_row']['n']}) "
                  f"peak={rec['peak_rss_mib']:.1f}MiB "
                  f"feat={rec['feat_ms_per_row']['median']*1e3:.1f}us/row", flush=True)
    print("\n[env]", json.dumps(res["_env"], ensure_ascii=False))


def report():
    res = json.load(OUT.open())
    print("=== bexp61 cost measurement v2 (GBDT library defaults; denominator = actual prediction rows; memory = subprocess peak MiB) ===")
    print(f"{'unit':<10}{'view':<8}{'dim':>4}{'fit_s':>8}{'lat1_ms':>10}{'p95':>8}"
          f"{'n':>4}{'us/row':>9}{'n':>4}{'peak MiB':>10}{'feat us/row':>13}")
    for c in res["cells"]:
        print(f"{c['unit']:<10}{c['view']:<8}{c['dim']:>4}{c['fit_s']:>8.2f}"
              f"{c['lat1_ms']['median']:>10.2f}{c['lat1_ms']['p95']:>8.2f}"
              f"{c['lat1_ms']['n']:>4}{c['thr_us_row']['median']:>9.1f}"
              f"{c['thr_us_row']['n']:>4}{c['peak_rss_mib']:>10.1f}"
              f"{c['feat_ms_per_row']['median']*1e3:>13.1f}")
    print("\n[env]", json.dumps(res["_env"], ensure_ascii=False))
    print("\nnote: single-row latency batch=1, repeats %d; throughput batch=%d, repeats %d; "
          "memory is the subprocess `ru_maxrss` peak (macOS bytes->MiB); "
          "feature cost = per-cell matrix construction time / retained rows (%d cells sampled)."
          % (REPEATS, BATCH, THR_REPEATS, 40))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
