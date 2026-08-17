"""bexp34: sensitivity to a missing anchor, and bexp35: the strict deployment mode
over all 11 units.

--- bexp34 (--missing) ---
Simulating early records lost in deployment: the first K0 cycles of every cell are
removed (training and test alike, labels with them), so the anchor falls back to
the earliest available cycle. K0 in {0, 10, 50, 100}; GBDT, 11 units x 3 seeds,
reference protocol. Cells left with fewer than 20 rows are skipped and counted.

================ frozen criteria (committed before the run) ================
M1: at K0=50 the median lift relative to raw is still negative (the direction
    holds); the degradation curve is written out
=======================================================

--- bexp35 (--strict) ---
The strict deployment pipeline extended to all 11 units: min/max fitted on the
source pool only (test rows are merely transformed), cycle_scaled (/200, no
dependence on lifetime), and conditions {raw, srl, history} x {GBDT 5 seeds,
TabPFN 3 seeds}.

================ frozen criteria (committed before the run) ================
S1: under the strict mode, either srl or history improves on raw in >= 7 of 11
    units for GBDT; TabPFN is an observation. A side-by-side table quantifying the
    loss against the reference protocol is written out
=======================================================
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import (CONTEXT_CAP, fit_predict_gbdt,
                                   fit_predict_tabpfn, subsample_context)
from battery_lab.temporal_features import (add_history_features,
                                           cycle_author_minmax, cycle_scaled)

OUT_M = ROOT / "results/battery/bexp34_missing_anchor.json"
OUT_S = ROOT / "results/battery/bexp35_strict_mode.json"
K0S = [0, 10, 50, 100]
MIN_ROWS = 20


def percell_norm(x):
    mn, mx = x.min(0, keepdims=True), x.max(0, keepdims=True)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return 2 * (x - mn) / rng - 1


def run_missing():
    state = json.load(OUT_M.open()) if OUT_M.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in [u for u, _, _ in FIT_UNITS]:
        unit = load_fit_unit(*umap[uid])
        for k0 in K0S:
            skipped = 0

            def prep(cells):
                nonlocal skipped
                out = []
                for c in cells:
                    if len(c.y) - k0 < MIN_ROWS:
                        skipped += 1
                        continue
                    out.append((percell_norm(c.X[k0:].astype(np.float64)),
                                c.y[k0:]))
                return out

            tr, te = prep(unit.train_cells), prep(unit.test_cells)
            if not tr or not te:
                state[f"{uid}|k{k0}"] = "SKIP: insufficient cells"
                continue
            for cond in ("raw", "srl"):
                key = f"{uid}|k{k0}|{cond}"
                if key in state:
                    continue
                xs, ys = [], []
                for Z, y in tr:
                    cols = [Z, cycle_author_minmax(len(Z))]
                    if cond == "srl":
                        cols.append(Z - Z[:1])
                    xs.append(np.hstack(cols)); ys.append(y)
                X_tr, y_tr = np.vstack(xs), np.concatenate(ys)
                xs, ys = [], []
                for Z, y in te:
                    cols = [Z, cycle_author_minmax(len(Z))]
                    if cond == "srl":
                        cols.append(Z - Z[:1])
                    xs.append(np.hstack(cols)); ys.append(y)
                X_te, y_te = np.vstack(xs), np.concatenate(ys)
                vals = []
                for seed in (0, 1, 2):
                    Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
                    pred = fit_predict_gbdt(Xc, yc, X_te, seed)
                    vals.append(float(np.sqrt(np.mean((y_te - pred) ** 2))))
                state[key] = {"rmse": float(np.mean(vals)), "skipped": skipped}
                json.dump(state, OUT_M.open("w"), indent=1)
        print(f"missing {uid} done", flush=True)
    report_missing(state)


def report_missing(state=None):
    state = state or json.load(OUT_M.open())
    print(f"\nmissing-anchor curve: median lift delta% by K0")
    for k0 in K0S:
        ds = []
        for uid, _, _ in FIT_UNITS:
            a = state.get(f"{uid}|k{k0}|raw"); b = state.get(f"{uid}|k{k0}|srl")
            if isinstance(a, dict) and isinstance(b, dict):
                ds.append((b["rmse"] - a["rmse"]) / a["rmse"] * 100)
        print(f"  K0={k0:<4} median delta%={np.median(ds):+.1f}  units better "
              f"{sum(1 for d in ds if d < 0)}/{len(ds)}")
    ds50 = [(state[f'{u}|k50|srl']['rmse'] - state[f'{u}|k50|raw']['rmse'])
            / state[f'{u}|k50|raw']['rmse'] * 100
            for u, _, _ in FIT_UNITS
            if isinstance(state.get(f'{u}|k50|raw'), dict)]
    ok = np.median(ds50) < 0
    print("M1 the median lift at K0=50 is negative: "
          + ("holds" if ok else "falsified"))


def strict_prepare(unit):
    pool = np.vstack([c.X for c in unit.train_cells]).astype(np.float64)
    mn, mx = pool.min(0, keepdims=True), pool.max(0, keepdims=True)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return {c.cell_id: 2 * (c.X.astype(np.float64) - mn) / rng - 1
            for c in unit.cells}


def run_strict():
    state = json.load(OUT_S.open()) if OUT_S.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in [u for u, _, _ in FIT_UNITS]:
        unit = load_fit_unit(*umap[uid])
        Xn = strict_prepare(unit)

        def build(cells, cond):
            xs, ys = [], []
            for c in cells:
                Z = Xn[c.cell_id]
                cols = [Z, cycle_scaled(len(Z))]
                if cond == "srl":
                    cols.append(Z - Z[:1])
                elif cond == "history":
                    cols.append(add_history_features(Z))
                xs.append(np.hstack(cols)); ys.append(c.y)
            return np.vstack(xs), np.concatenate(ys)

        for cond in ("raw", "srl", "history"):
            for bk, fit, seeds in (("gbdt", fit_predict_gbdt, [0, 1, 2, 3, 4]),
                                   ("tabpfn", fit_predict_tabpfn, [0, 1, 2])):
                key = f"{uid}|{cond}|{bk}"
                if key in state:
                    continue
                X_tr, y_tr = build(unit.train_cells, cond)
                X_te, y_te = build(unit.test_cells, cond)
                t0 = time.time()
                vals = []
                for seed in seeds:
                    Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
                    pred = fit(Xc, yc, X_te, seed)
                    vals.append(float(np.sqrt(np.mean((y_te - pred) ** 2))))
                state[key] = float(np.mean(vals))
                json.dump(state, OUT_S.open("w"), indent=1)
                print(f"strict {key:<28} rmse={state[key]:.5f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    report_strict(state)


def report_strict(state=None):
    state = state or json.load(OUT_S.open())
    print(f"\nstrict deployment mode (source-side normalisation + scaled cycle) "
          f"RMSE")
    print(f"{'unit':<16}{'G-raw':>9}{'G-srl':>9}{'G-hist':>9}{'T-raw':>9}"
          f"{'T-srl':>9}{'T-hist':>9}")
    n_srl = n_hist = 0
    for uid, _, _ in FIT_UNITS:
        row = f"{uid:<16}"
        for bk in ("gbdt", "tabpfn"):
            for cond in ("raw", "srl", "history"):
                v = state.get(f"{uid}|{cond}|{bk}")
                row += f"{v:>9.5f}" if isinstance(v, float) else f"{'--':>9}"
        print(row)
        g0 = state.get(f"{uid}|raw|gbdt")
        if isinstance(g0, float):
            if state.get(f"{uid}|srl|gbdt", 9e9) < g0:
                n_srl += 1
            if state.get(f"{uid}|history|gbdt", 9e9) < g0:
                n_hist += 1
    ok = max(n_srl, n_hist) >= 7
    print(f"\nS1 GBDT: srl better on {n_srl}/11, history better on {n_hist}/11 "
          f"(criterion max>=7) -> " + ("holds" if ok else "falsified"))


if __name__ == "__main__":
    if "--missing" in sys.argv:
        run_missing()
    elif "--strict" in sys.argv:
        run_strict()
    else:
        report_missing()
        report_strict()
