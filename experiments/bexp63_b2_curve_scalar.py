"""bexp63: F03-B2 -- comparison against the real curve-difference scalar feature on the same SOH
task (main matrix).

Per `f03_baseline_completion_plan.md` section 5-B (B2):
  Combine the cell-level scalar derived from the **real curves** (Severson's log-variance of
  dQ(V), from bexp62) with the fixed-scale cycle position as features known after cycle 100; train
  GBDT on the source side (train cells), keep the official splits unchanged, and compare on the
  same scoring set **after cycle 100**. Contrasting arms (same scoring set, same model, same seed
  set):

    cyc             : fixed-scale cycle position only (1 dim)
    cyc+scalar      : cycle position + dQ(V) scalar (2 dims; the scalar standardized with the
                      **source-side** mean/variance)
    raw             : released 16 statistics + cycle position (17 dims)
    srl             : compact SRL (33 dims = bexp50's srl view)
    history         : the full history stack (81 dims)

  Note: this table answers **only** "is B2's curve-difference scalar stronger than the cycle
  position under a common protocol, and how does it compare with the released features"; it does
  not extrapolate to R1-M5's other tasks (the original lifetime-prediction task is B1, the
  physical reference is bexp64/C).

================== criteria (frozen before the run) ==================
B2-1 scoring set: each test cell keeps only rows with retained-record index (0-based) >= 100;
             **all arms use the same set** (defined by the cyc arm's available rows; the other
             arms take the intersection). If row sets differ between arms, the latter is void.
B2-2 scalar standardization: the dQ(V) scalar is standardized with the **source-side (train
             cells)** mean/std; test-side statistics are forbidden.
B2-3 coverage: if a test cell has no scalar in bexp62 (insufficient cycles / missing curve), that
             cell is dropped from **all arms** together, and the drop list is reported (the cost
             is shared by all arms; the scalar arm must not pay alone).
B2-4 verdict bar: "the curve-difference scalar brings a uniform improvement" may be written only
             when `cyc+scalar` improves >= 7/11 units over `cyc` and the median delta > 0;
             otherwise the text reports only the observed direction and the unit count.
B2-5 re-run: results are written to `results/battery/bexp63_b2_curve_scalar.jsonl` (append-only,
             deduplicated keys); existing keys are not recomputed; `--report` only prints the
             archived statistics.
==================================================================

================== criteria v2 (2026-09-23, driven by third-round review F3-02) ==================
B2-6 exclusion: cells failing identity/mapping in manifest.release_gate (identity_pass=False) must
             be excluded from training and scoring **as a whole** (not only from the scoring
             side); each unit writes the excluded cells into the result field
             `excluded_identity` (with reason). The v1 archive let the two R3/RW batches (16 cells,
             all in release_gate.mapping_failures) participate as normal cells -- corrected in the
             v2 re-run; the remaining units have the same cells_ok under v2 as under v1 (the
             satellite batch has no scalar and is excluded in both versions).
B2-7 availability: the check of "raw cycle 100" availability for scoring rows lives in bexp65
             (depends on the manifest and bexp62 output); **bexp65's violations (incomplete
             scoring-row mapping / raw cycle <= 100) are also excluded from B2 as a whole** (same
             mechanism as B2-6, reason recorded as availability_violation). The scoring set of
             this script remains retained-record index >= 100 (independent of the row mapping
             itself).
==================================================================

Run:  ./.venv/bin/python experiments/bexp63_b2_curve_scalar.py            # full matrix
      ./.venv/bin/python experiments/bexp63_b2_curve_scalar.py --report
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

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.temporal_features import build_view

OUT = ROOT / "results/battery/bexp63_b2_curve_scalar.jsonl"
CURVE = ROOT / "results/battery/bexp62_curve_features.json"
READER = "strict_online"
N0 = 100                     # scoring start: retained-record index >= 100
SEEDS = (0, 1, 2, 3, 4)
CONTEXT_CAP = 2048
ARMS = ("cyc", "cyc+scalar", "raw", "srl", "history")


def _load_bexp50():
    spec = importlib.util.spec_from_file_location(
        "bexp50", ROOT / "experiments" / "bexp50_linear_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B50 = _load_bexp50()


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["unit"], r["arm"], r["seed"]))
    return keys


def load_excluded():
    """B2 exclusion set: release_gate (fail_identity/unverifiable) + bexp65 availability violations (B2-6/B2-7)."""
    ex = {}
    p = ROOT / "results/battery/raw_data_manifest.json"
    if p.exists():
        man = json.load(open(p))
        gate = man.get("release_gate") or {}
        for key in ("identity_failures", "unverified"):
            for r in gate.get(key, []):
                ex[r["cell"]] = r["status"]
    else:
        print("[bexp63] warning: raw_data_manifest.json missing; the B2-6 exclusion list is unavailable")
    p65 = ROOT / "results/battery/bexp65_b2_availability.json"
    if p65.exists():
        a65 = json.load(open(p65))
        for r in a65.get("violations", []):
            ex[r["cell"]] = "availability_violation"
    else:
        print("[bexp63] warning: bexp65 output missing; the B2-7 availability exclusion list is unavailable")
    return ex


def _scalar_stats(scalars_tr):
    v = np.array([s for s in scalars_tr if s is not None], float)
    return float(v.mean()), float(v.std() + 1e-12)


def eval_unit(unit, arm, seed, curve, cells_ok):
    tr = [c for c in unit.train_cells if c.cell_id in cells_ok]
    te = [c for c in unit.test_cells if c.cell_id in cells_ok]
    if not tr or not te:
        return None
    Xn = normalize_cells(tr + te, "source", source_cells=tr)
    cycle_mode = "scaled200"

    def arm_matrix(cells, scalars_std=None):
        xs, ys, ids = [], [], []
        for c in cells:
            base = build_view(Xn[c.cell_id], "raw", cycle_mode=cycle_mode)   # 17 dims
            cyc = base[:, -1:]
            Z = Xn[c.cell_id]
            if arm == "cyc":
                X = cyc
            elif arm == "cyc+scalar":
                s = curve.get(c.cell_id, {}).get("qdiff_var_log10")
                if s is None:
                    return None
                z = (s - scalars_std[0]) / scalars_std[1]
                X = np.hstack([cyc, np.full((len(cyc), 1), z)])
            else:
                X = B50.build_xy(Z, arm, cycle_mode)
            # scoring set: retained-record index >= N0 (the rows of this arm must match the cyc
            # arm -> use the same row mask)
            pos = np.arange(len(c.y))
            m = pos >= N0
            xs.append(X[m]); ys.append(c.y[m]); ids.extend([c.cell_id] * int(m.sum()))
        return np.vstack(xs), np.concatenate(ys), np.array(ids)

    # source-side scalar statistics (train cells only; B2-2)
    scalars_tr = [curve.get(c.cell_id, {}).get("qdiff_var_log10") for c in tr]
    s_mean, s_std = _scalar_stats(scalars_tr)
    out_tr = arm_matrix(tr, (s_mean, s_std))
    out_te = arm_matrix(te, (s_mean, s_std))
    if out_tr is None or out_te is None:
        return None
    X_tr, y_tr, _ = out_tr
    X_te, y_te, te_ids = out_te
    X_tr, y_tr = B50.subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
    pred = B50.MODELS["gbdt"](X_tr, y_tr, X_te, seed)
    per_cell = {cid: float(np.sqrt(np.mean((y_te[te_ids == cid] - pred[te_ids == cid]) ** 2)))
                for cid in np.unique(te_ids)}
    return {"unit": unit.unit_id, "arm": arm, "seed": seed, "reader": READER,
            "n_train_rows": int(len(y_tr)), "n_test_rows": int(len(y_te)),
            "n_test_cells": int(len(per_cell)),
            "rmse_overall": float(np.sqrt(np.mean((y_te - pred) ** 2))),
            "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
            "per_cell": per_cell}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--units", nargs="+", default=None)
    a = ap.parse_args()

    curve = {k: v for k, v in json.load(open(CURVE))["cells"].items()}
    if a.report:
        rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
        return report(rows)

    keys = done_keys()
    excluded = load_excluded()
    units = [u for u in FIT_UNITS if a.units is None or u[0] in a.units]
    t0 = time.time()
    with OUT.open("a") as f:
        for uid, ds, bk in units:
            unit = load_fit_unit(ds, bk, reader=READER)
            # B2-3: test cells lacking the scalar are dropped from **all arms**; B2-6: release_gate
            # failures are excluded as a whole
            cells_ok = {c.cell_id for c in unit.cells
                        if curve.get(c.cell_id, {}).get("qdiff_var_log10") is not None
                        and c.cell_id not in excluded}
            excluded_here = sorted(c.cell_id for c in unit.cells if c.cell_id in excluded)
            dropped = [c.cell_id for c in unit.test_cells if c.cell_id not in cells_ok]
            if not any(c.cell_id in cells_ok for c in unit.test_cells):
                print(f"[skip] {uid}: test cells all lack the curve scalar (B2-3 drop-all), dropped={dropped}")
                for arm in a.arms:      # explicitly log the skip; never silently
                    f.write(json.dumps({"unit": uid, "arm": arm, "seed": None,
                                        "status": "skipped_no_scalar",
                                        "excluded_identity": excluded_here,
                                        "dropped_test_cells": dropped},
                                       ensure_ascii=False) + "\n")
                f.flush()
                continue
            for arm in a.arms:
                for seed in SEEDS:
                    if (uid, arm, seed) in keys:
                        continue
                    r = eval_unit(unit, arm, seed, curve, cells_ok)
                    if r is None:
                        print(f"[skip] {uid} {arm} seed={seed} (empty matrix)")
                        continue
                    r["dropped_test_cells"] = dropped
                    r["excluded_identity"] = excluded_here
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{uid}] {arm:12s} seed={seed} macro={r['rmse_cell_macro']:.6f} "
                          f"cells={r['n_test_cells']} rows={r['n_test_rows']}")
    print(f"[bexp63] took {time.time()-t0:.0f}s -> {OUT}")
    return report([json.loads(l) for l in OUT.read_text().splitlines() if l.strip()])


def report(rows):
    rows = [r for r in rows if "rmse_cell_macro" in r]     # drop skip-marker rows
    skips = [r for r in rows if False]
    print("\n== B2 main matrix (per-unit median over 5 seeds -> median across units) ==")
    units = sorted({r["unit"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    hdr = "unit".ljust(10) + "".join(a.rjust(13) for a in arms)
    print(hdr)
    med = {}
    for u in units:
        line = u.ljust(10)
        for a in arms:
            v = [r["rmse_cell_macro"] for r in rows if r["unit"] == u and r["arm"] == a]
            m = float(np.median(v)) if v else float("nan")
            med[(u, a)] = m
            line += f"{m:13.6f}" if v else " " * 13
        print(line)
    print("-" * len(hdr))
    print("median".ljust(10) + "".join(
        f"{np.median([med[(u, a)] for u in units if not np.isnan(med[(u, a)])]):13.6f}"
        if any(not np.isnan(med[(u, a)]) for u in units) else " " * 13 for a in arms))
    # B2-4: per-unit improvement of cyc+scalar vs cyc
    if ("cyc+scalar" in arms) and ("cyc" in arms):
        better = 0
        deltas = []
        for u in units:
            d = med[(u, "cyc")] - med[(u, "cyc+scalar")]
            deltas.append(d)
            better += d > 0
        print(f"\nB2-4: units improved for cyc+scalar vs cyc = {better}/{len(units)}, "
              f"median delta = {np.median(deltas):+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
