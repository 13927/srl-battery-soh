"""bexp64: F03-C -- the full charge-segment I dt integral as a **physical reference** (not a
label identity).

Per `f03_baseline_completion_plan.md` section 6-C:
  Build one physicalized simple reference from the real curves: normalize the "charge-segment
  current integral of cycle k" by the rated capacity, calibrate the efficiency eta on the
  **source side (train cells)** (median ratio y / (I dt / Qnom), using only the first 100
  cycles), then give a per-cycle SOH estimate and compare it on the same scoring set after
  cycle 100.

Key discipline (explicit in plan section 6-C):
  - the label is on the **discharge** side (`capacity / nominal`); this reference uses the
    **charge**-side integral plus a source-calibrated eta, so it does not recompute the label
    (no label identity). If a cell's charge/discharge sides cannot be separated, record `skip`.
  - eta is calibrated on the **source side (train cells)** first 100 cycles only; the test side
    is never touched.
  - explicit coverage: scoring is row-by-row only on cells with "row-level alignment"
    (manifest.align.row_level_mapping=True); other cells are recorded as `no_row_mapping` and
    dropped from this reference (their cost is not charged to the other methods).

================== criteria (frozen before the run) ==================
C1 scoring set: same as bexp63 -- retained-record index (0-based) >= 100; all comparison objects
   share the same row set.
C2 eta calibration: eta = median over {source-side cells, first 100 cycles} of [ y / (q_chg/Qnom) ];
   cycles with a zero denominator are dropped; if the calibration sample has < 50 rows, record
   `insufficient_calibration` and skip that unit.
C3 reference definition: yhat_k = eta x q_chg(k) / Qnom (q_chg is the raw charge-segment I dt
   integral in Ah; Qnom is the unit's rated capacity).
C4 reporting: per unit, report the median per-cell RMSE of "this reference / the cyc arm of
   bexp63 (same scoring set)", the number of covered cells and the drop list; no "better than"
   verdict, only observations.
C5 re-run: output `results/battery/bexp64_physical_ref.jsonl` (unique per unit+seed).
==================================================================

================== criteria v2 (2026-09-23, driven by third-round review F3-01/F3-03) ==================
Background: v1 joined scoring rows to raw curves with "continuous position pos = arange(len(y)) +
offset". After strict_online drops rows with invalid inputs, pos is no longer the CSV row number,
and the row-to-raw-cycle join is globally misaligned (measured: for two XJTU-2C test cells,
110/262 and 128/288 rows in the scoring range have positions differing from the saved row numbers,
by up to 22 rows).
C1' row mapping: scoring rows must take values through the manifest's **explicit mapping**
   `align.raw_index_of_proc` (CSV row -> raw cycle): csv_row = record_index[pos]; raw_cycle =
   map[csv_row]. If any scoring row is unmappable (-1/out of range), the cell is dropped as a
   whole and logged (keeping the row set complete).
C1'' availability: per unit, report the raw-cycle range (min/max) of the scoring rows; the
   relation of min raw cycle to 101 is written to the output (the scalar-availability check is
   expanded on the B2 side in bexp65).
C2' source calibration: only source-side rows with "raw cycle number < 100" are used
   (interpreting N=100 through the raw cycle ID, plan section 3), and rows must be mappable;
   unmappable rows are dropped and counted.
C3' units (F3-03 + fourth-round A): XJTU relative_time_min x 60, current_A in amperes; MIT t in
   minutes, I is relative current (1C = 1.1 Ah) -- q_chg_Ah = Qnom x sum over positive segments
   of I dt, done by `bexp62.positive_segment_ah` on the **complete arrays**.
   v1 missed the Qnom multiplier; fixed in v2; v3 switched to "sum over segments" but
   **introduced an index-space misalignment** (slicing the compacted array with the complete
   mask's segment boundaries: segments after the first were mis-sliced/lost, synthetic
   two-segment sample -66.7%, confirmed by fourth-round audit item A); v5 (2026-09-23b) fixed it
   via the shared function and accepts it on synthetic samples (two segments / leading rest /
   single segment / all non-positive) in tests/test_curve_units.py; v4's data-quality gate is
   retained (cycle duration span > 2 h -> NaN -- a **post-diagnostic revision**). True closure
   and RMSE follow the re-run archives.
C4' contrast: same-set contrast -- the comparison arm is the cyc arm of bexp63 on the **same
   scoring-cell subset** of that unit (the bexp63 row set = retained-record index >= 100, row-wise
   identical to this reference).
C5' audit: per-scored-point output `results/battery/bexp64_scored_points.jsonl`
   (cell_id, csv_row, raw_cycle, q_chg, y, yhat).
C6 exclusion: fail_identity/unverifiable cells from manifest.release_gate are excluded from
   training (calibration) and scoring as a whole (same list as bexp63); fail_mapping cells have
   usable identity but cannot be read row by row, and are filtered on the scoring side by the
   row_level_mapping check.
==================================================================

Run:  ./.venv/bin/python experiments/bexp64_physical_ref.py
      ./.venv/bin/python experiments/bexp64_physical_ref.py --report
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit

OUT = ROOT / "results/battery/bexp64_physical_ref.jsonl"
OUT_POINTS = ROOT / "results/battery/bexp64_scored_points.jsonl"
MANIFEST = ROOT / "results/battery/raw_data_manifest.json"
READER = "strict_online"
N0 = 100
NOMINAL = {"XJTU": 2.0, "TJU": 3.5, "HUST": 2.9, "MIT": 1.1}
XJTU_NOMINAL = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0}   # Batch-1..6
TJU_NOMINAL = {0: 3.5, 1: 3.5, 2: 2.5}
# XJTU batch keys (battery_lab.data_adapters.XJTU_BATCHES) -> extracted directory names
XJTU_BATCH_DIR = {"2C": "Batch-1", "3C": "Batch-2", "R2.5": "Batch-3",
                  "R3": "Batch-4", "RW": "Batch-5", "satellite": "Batch-6"}


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B62 = _load("bexp62", ROOT / "experiments" / "bexp62_curve_features.py")


def qchg_series_xjtu(path):
    import scipy.io as sio
    mat = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    data = np.atleast_1d(mat["data"])
    out = []
    for c in data:
        rt = np.asarray(getattr(c, "relative_time_min", []), float)
        i = np.asarray(getattr(c, "current_A", []), float)
        if rt.size < 4:
            out.append(np.nan); continue
        s = B62.longest_run(i > 0)
        out.append(B62.trapz_ah(rt[s] * 60.0, i[s]))
    return np.array(out, float)


def qchg_series_mit(path, order):
    """MIT per-cycle charge integral: qchg_Ah = Qnom x sum over positive segments of I dt (complete-array segmentation; shared with bexp62).

    v5 (2026-09-23b, fourth-round audit item A): v3/v4 used the complete mask's segment boundaries
    to slice the compacted (t[m], i[m]) -- the index-space misalignment mis-sliced/lost segments
    after the first (synthetic two-segment sample 0.01833 vs 0.055 Ah, -66.7%); this version calls
    `B62.positive_segment_ah(t, i)` directly to integrate segment by segment on the complete
    arrays. v4's data-quality gate is retained: cycle duration span > 2 h (7200 s) -> NaN (normal
    ~54 min; a **post-diagnostic revision**). t is in minutes (x60 -> seconds); I is relative
    current (1C=1.1 Ah).
    """
    import h5py
    with h5py.File(str(path), "r") as f:
        b = f["batch"]
        cyc = f[b["cycles"][order, 0]]
        n = int(cyc["V"].shape[0])
        out = []
        for k in range(n):
            i = np.asarray(f[cyc["I"][k, 0]][()]).ravel().astype(float)
            if i.size == 0:
                out.append(np.nan); continue
            t = np.asarray(f[cyc["t"][k, 0]][()]).ravel().astype(float) * 60.0 if "t" in cyc \
                else np.arange(i.size, dtype=float)
            m = i > 0
            if not m.any():
                out.append(np.nan); continue
            if float(t[m].max() - t[m].min()) > 7200.0:   # v4 data-quality gate
                out.append(np.nan); continue
            tot = B62.positive_segment_ah(t, i)           # v5: complete-array segment integration
            out.append(np.nan if tot is None else tot * NOMINAL["MIT"])
        return np.array(out, float)


def build_qchg(units, raw_xjtu, raw_mit):
    """Build the charge-integral series per cell; return {cell_id: np.ndarray}."""
    out = {}
    for uid, ds, bk in units:
        if ds == "XJTU":
            folder = XJTU_BATCH_DIR.get(bk)
            if folder is None:
                print("[skip]", uid, f"XJTU batch key {bk!r} has no directory mapping")
                continue
            cell_ids = sorted(p.stem for p in (raw_xjtu / folder).glob("*.mat"))
            for cid in cell_ids:
                out[cid] = qchg_series_xjtu(raw_xjtu / folder / f"{cid}.mat")
        elif ds == "MIT":
            man = json.load(open(MANIFEST))
            for batch, meta in man["mit"].items():
                fname = meta["raw_file"].split("/")[-1]
                for c in meta["cells"]:
                    cid = f"{batch}_battery-{c['order_index']}"
                    out[cid] = qchg_series_mit(Path(raw_mit) / fname, c["order_index"] - 1)
    return out


def unit_nominal(ds, bk):
    if ds == "XJTU":
        return XJTU_NOMINAL.get(bk, NOMINAL["XJTU"])
    if ds == "TJU":
        return TJU_NOMINAL.get(bk, NOMINAL["TJU"])
    return NOMINAL[ds]


def load_maps():
    """manifest -> {cell_id: align dict} and the identity-failure set (release_gate)."""
    man = json.load(open(MANIFEST))
    maps, excluded = {}, set()
    for cell, v in man["xjtu"].items():
        al = v.get("align")
        if al:
            maps[cell] = al
    for b, grp in man["mit"].items():
        for c in grp["cells"]:
            al = c.get("align")
            if al:
                maps[f"{b}_battery-{c['order_index']}"] = al
    gate = man.get("release_gate") or {}
    # C6/B2 consistency: exclude fail_identity and unverifiable (same list as bexp63);
    # fail_mapping cells have usable identity but are not row-readable -- filtered on the scoring
    # side by the row_level_mapping check.
    for key in ("identity_failures", "unverified"):
        for r in gate.get(key, []):
            excluded.add(r["cell"])
    return maps, excluded


def raw_cycles_of_rows(c, al, rows_mask):
    """Map retained rows (selected by rows_mask) to raw cycle numbers; return (csv_rows, raw_cycles)."""
    ridx = np.asarray(c.record_index)
    raw = np.asarray(al["raw_index_of_proc"], dtype=int)
    csv_rows = ridx[rows_mask]
    inb = (csv_rows >= 0) & (csv_rows < raw.size)
    rc = np.full(len(csv_rows), -1, dtype=int)
    rc[inb] = raw[csv_rows[inb]]
    return csv_rows, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-xjtu", default=str(Path.home() / "data_raw/xjtu/xjtu_raw/Battery Dataset"))
    ap.add_argument("--raw-mit", default=str(Path.home() / "data_raw/mit"))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report:
        rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
        return report(rows)

    maps, excluded = load_maps()
    qchg = build_qchg(FIT_UNITS, Path(a.raw_xjtu).expanduser(), Path(a.raw_mit).expanduser())
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["unit"])
    with OUT.open("a") as f, OUT_POINTS.open("a") as fp:
        for uid, ds, bk in FIT_UNITS:
            if uid in done:
                continue
            unit = load_fit_unit(ds, bk, reader=READER)
            qnom = unit_nominal(ds, bk)
            # ---- source-side calibration (C2'): only source rows with raw cycle < N0, mappable ----
            ratios, calib_rows, calib_dropped = [], 0, 0
            for c in unit.train_cells:
                if c.cell_id in excluded:
                    continue
                al = maps.get(c.cell_id)
                q = qchg.get(c.cell_id)
                if al is None or q is None or not al.get("identity_pass"):
                    continue
                pos = np.arange(len(c.y))
                _, rc_all = raw_cycles_of_rows(c, al, np.ones(len(c.y), bool))
                m = (rc_all >= 0) & (rc_all < N0) & np.isfinite(q[np.maximum(rc_all, 0)])
                calib_rows += int(m.sum())
                calib_dropped += int((rc_all >= 0).sum() - int(m.sum()))
                if m.sum():
                    qq = q[rc_all[m]] / qnom
                    ok = np.isfinite(qq) & (qq > 0)
                    ratios.extend((c.y[m][ok] / qq[ok]).tolist())
                del pos
            if len(ratios) < 50:
                print(f"[skip] {uid}: insufficient_calibration n={len(ratios)}")
                continue
            eta = float(np.median(ratios))
            # ---- test-side per-cycle scoring (C1'/C3'/C4') ----
            per_cell, dropped, points = {}, [], []
            scored_rc = []
            for c in unit.test_cells:
                if c.cell_id in excluded:
                    dropped.append(c.cell_id + ":identity_failed")
                    continue
                al = maps.get(c.cell_id)
                q = qchg.get(c.cell_id)
                if al is None or q is None or not al.get("row_level_mapping"):
                    dropped.append(c.cell_id + ":no_row_mapping")
                    continue
                pos = np.arange(len(c.y))
                rows_mask = pos >= N0
                csv_rows, rc = raw_cycles_of_rows(c, al, rows_mask)
                n_invalid = int((rc < 0).sum())
                if n_invalid or rc.size == 0:
                    dropped.append(f"{c.cell_id}:unmapped_rows={n_invalid}")
                    continue
                qv = q[rc]
                ok = np.isfinite(qv)
                if not ok.all():
                    dropped.append(f"{c.cell_id}:qchg_nan={int((~ok).sum())}")
                    continue
                y = c.y[rows_mask]
                yhat = eta * qv / qnom
                per_cell[c.cell_id] = float(np.sqrt(np.mean((y - yhat) ** 2)))
                scored_rc.extend(rc.tolist())
                pts = [{"cell_id": c.cell_id, "csv_row": int(r), "raw_cycle": int(k),
                        "q_chg_Ah": float(p), "y": float(yy), "yhat": float(hh)}
                       for r, k, p, yy, hh in zip(csv_rows, rc, qv, y, yhat)]
                points.append(pts)
            if not per_cell:
                print(f"[skip] {uid}: no row-level-alignable test cells")
                continue
            r = {"unit": uid, "eta": eta, "calib_rows": calib_rows,
                 "calib_dropped_rows": calib_dropped, "n_test_cells_scored": len(per_cell),
                 "dropped_test_cells": dropped,
                 "scored_rows": int(len(scored_rc)),
                 "raw_cycle_range": [int(min(scored_rc)), int(max(scored_rc))],
                 "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
                 "per_cell": per_cell}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            fp.write(json.dumps({"unit": uid, "points": [p for sub in points for p in sub]},
                                ensure_ascii=False) + "\n")
            fp.flush()
            print(f"[{uid}] eta={eta:.4f} macro={r['rmse_cell_macro']:.6f} "
                  f"cells={len(per_cell)} rows={r['scored_rows']} "
                  f"raw_cyc=[{r['raw_cycle_range'][0]},{r['raw_cycle_range'][1]}] dropped={len(dropped)}")
    return report([json.loads(l) for l in OUT.read_text().splitlines() if l.strip()])


def report(rows):
    print("\n== C physical reference (charge I dt + source-side eta; v2 explicit row mapping) per unit ==")
    for r in rows:
        print(f"{r['unit']:10s} eta={r['eta']:.4f} macro={r['rmse_cell_macro']:.6f} "
              f"cells={r['n_test_cells_scored']} rows={r.get('scored_rows')} "
              f"dropped={r['dropped_test_cells']}")
    if rows:
        print(f"\nmedian macro across units = {np.median([r['rmse_cell_macro'] for r in rows]):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
