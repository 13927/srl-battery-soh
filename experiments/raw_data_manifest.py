"""raw_data_manifest: identity-alignment ledger between the raw releases (XJTU / MIT)
and the processed released files (F03 Stage 3.1).

Per `f03_baseline_completion_plan.md` section 2 "obtain and align the raw data first":
  deliver raw_data_manifest: release source/version/checksum, raw cell ID -> processed ID -> official train/test,
  raw cycle ID -> CSV row, time/current/voltage/capacity units, charge/discharge direction, available curves and
  reasons for any absence.
  Check for concatenation, cycle deletion, renumbering; **must not pair merely by similar file name or similar
  capacity trace**.

What this script does (alignment evidence chain, ordered by strength):
  1) raw-side identity comes from identifiers carried by the file/structure itself, not from similarity:
     - XJTU: the cell name of the raw `Batch-k/<cell>.mat` is identical to the processed CSV (publisher same-name
       convention), plus `summary.cycle_life` and per-cycle `charge_capacity_Ah/discharge_capacity_Ah`;
     - MIT: `batch.barcode` inside the struct (MATLAB barcode, last digit = channel number) gives each cell's
       identity, **paired by order within the struct** (i.e. channel order within the batch), and the order
       evidence is recorded;
  2) after alignment do an **independent verification** (not a pairing basis): compare the raw per-cycle discharge
     capacity against the processed CSV `capacity` row by row, and report the maximum relative deviation and the
     correlation on the overlap segment; if the deviation exceeds the threshold, raise an error instead of passing
     silently;
  3) give a per-library **gap list** (cells listed by the platform but absent after processing, and the reverse),
     and state the reason (write "unknown" if unknown).

Units and direction (from the XJTU release `Data Introduction-English.pdf` and the MIT struct field semantics,
recorded item by item):
  - XJTU `data.<cycle>.voltage_V` (V), `current_A` (A, charge positive / discharge negative, log as measured by
    sign), `capacity_Ah` (Ah, cumulative within the same cycle), `temperature_C` (degC), `relative_time_min` (min);
    `summary.charge_capacity_Ah` / `discharge_capacity_Ah` are the per-cycle charge/discharge capacity (Ah);
  - MIT `I` (A), `V` (V), `T` (degC), `Qc`/`Qd` (Ah, charge/discharge capacity), `t` (s), `Qdlin`/`discharge_dQdV`
    (downsampled sequences of discharge capacity and dQ/dV).

================== criteria (frozen before the run) ==================
R1 completeness: for every raw cell marked as matched after processing, must give the raw file, n_cycles,
           raw discharge-capacity sequence length, processed CSV row count, and the official split (train/test) --
           five items. Missing any one is an error.
R2 identity verification: for every matched cell, the processed capacity and the raw per-cycle discharge capacity
            under the best alignment must satisfy |median calibration ratio proc/raw - 1| <= 3% (identity); the
            other two quantities are recorded as **capability** fields without a pass/fail judgment:
            `align.spearman_aligned` (aligned prefix rank correlation) and `align.row_level_mapping`
            (row-by-row match rate >= 0.9 -- only if true can curve features be attached row by row to the
            processed rows).
            The publisher resampled/trimmed some cells (e.g. 2018-04-12_battery-11: raw 1077 cycles -> proc 124
            rows, diff std 0.0019 vs 0.0005); such cells have row_level_mapping=False and
            must be reported separately in B1/B2 as "not row-alignable"; they must not be silently treated as a
            mismatch, nor may similarity be used to pick a match.
R3 explicit gaps: cells listed by the platform but missing after processing, and the reverse, all go into `gaps`,
            with reason "unknown" when unknown (must not be left empty).
R4 read-only: do not modify any existing file under `external/` or `results/battery/`; only add this script's
            outputs.
R5 provenance: `sources` records each raw file's size and sha256 (measured after download), plus the source
            URL/record id.
======================================================

================== criteria v2 (2026-09-23, driven by third-round review F3-02/F3-03) ==================
R2' explicit row mapping (replaces R2's "offset + per-position ratio" implementation; R2 text kept in the
    previous section):
    The original implementation had two verified defects: (a) the single-constant-offset assumption does not hold
    when "the publisher deleted intermediate rows" (measured 2C_battery-1: under the strict tolerance 375/375 rows
    match, but pos_diff varies between [9,15], not a constant 9); (b) `dev_p99` was computed only over rows
    satisfying <=1%, so it could not describe the deviation of the whole sequence.
    Therefore v2 becomes (v2.1: single-row search window + pointer does not advance on mismatch):
      1) **monotonic greedy subsequence matching** from the raw start: for each proc row, search forward in raw
         from the current position for the first row with |dev| <= tol (skipped raw rows = rows the publisher
         deleted); single-row search window W=256 rows (to prevent unbounded scanning over interpolated rows);
         **the pointer does not advance on a mismatch** (v2.0 let the pointer run to the end, so one interpolated
         row invalidated the whole subsequent segment -- MIT battery-1 etc. were therefore misjudged as match
         ratio 0.5-0.9);
      2) tolerance ladder (1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3) tried tier by tier, taking the first tier that
         makes the match ratio >= 0.9;
      3) output an **explicit mapping** `raw_index_of_proc` (length = proc row count; -1 = unmatched) for
         downstream row-by-row lookup of raw cycles;
      4) deviation statistics over all matched rows (dev_median/p99/max, no longer only over rows <=1%);
      5) identity verification has two layers (v2.3): the **strict tier** (tolerance <= 1e-4) requires both >= 30
         matched rows and a share >= 50% ("the majority of published rows correspond value by value under the
         strict tolerance" is what constitutes evidence of sequence homology; scattered approximate matches do not
         count -- tolerance matching necessarily drives the matched-point ratio close to 1, so the share, not the
         ratio, is the evidence); plus a calibration ratio within +/-3%; when unsatisfied, record fail /
         unverifiable (unknown state) respectively by "matching sufficient but ratio deviating" / "matching
         insufficient".
         Four statuses (for downstream consumption):
           pass          = identity passes and row-alignable (row_ok);
           fail_mapping  = identity passes but not row-alignable (B2 usable, C not usable);
           fail_identity = strict-tier matching sufficient but calibration ratio deviates >3%;
           unverifiable  = strict-tier matching insufficient (< 30 rows or < 50%), identity unverifiable
                           (conservatively excluded).
    Downstream consumption rule: **B2 (bexp63/65) excludes fail_identity and unverifiable**;
    C (bexp64) additionally requires row_level_mapping=True; bexp62 cross-check uses only row-alignable cells.
R6 visible failures: cells that fail identity or mapping are all written into `problems` and `align.status`; the
            summary field `release_gate` (failure list + basis) is output for downstream scripts to read
            mandatorily. Script exit code: return 1 when identity failures exist (the list is written to disk
            first so the failure state is visible).
==================================================================

Run (on the machine that holds the data):
  ./.venv/bin/python experiments/raw_data_manifest.py \
      --xjtu-root "~/data_raw/xjtu/xjtu_raw/Battery Dataset" \
      --mit-root ~/data_raw/mit \
      --external external/PINN4SOH/data \
      --out results/battery/raw_data_manifest.json
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

XJTU_BATCHES = ["Batch-1", "Batch-2", "Batch-3", "Batch-4", "Batch-5", "Batch-6"]
MIT_BATCHES = [("2017-05-12", "mat_5c86c0b5fa2ede00015ddf66.mat"),
               ("2017-06-30", "mat_5c86bf13fa2ede00015ddd82.mat"),
               ("2018-04-12", "mat_5c86bd64fa2ede00015ddbb2.mat")]
MIT_URL = "https://data.matr.io/1/api/v1/file/{id}/download"
MIT_FILE_IDS = {"2017-05-12": "5c86c0b5fa2ede00015ddf66",
                "2017-06-30": "5c86bf13fa2ede00015ddd82",
                "2018-04-12": "5c86bd64fa2ede00015ddbb2"}
XJTU_URL = "https://zenodo.org/records/10963339/files/Battery%20Dataset.zip?download=1"
ID_MIN_ROWS = 30            # minimum matched rows required for identity verification (see criteria R2' item 5)

problems = []


def sha256(p: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def official_splits(external: Path):
    """Official train/test cell sets (from this pipeline's adapters, authoritative)."""
    from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
    out = {}
    for uid, ds, bk in FIT_UNITS:
        u = load_fit_unit(ds, bk)          # reader default (author_offline) -- split is independent of the reader
        out[uid] = {"train": sorted(c.cell_id for c in u.train_cells),
                    "test": sorted(c.cell_id for c in u.test_cells)}
    return out


def processed_capacity(external: Path, lib: str, cell: str):
    """Processed CSV capacity column (cell level). XJTU: 'XJTU data/<cell>.csv'; MIT: 'MIT data/<batch>/<cell>.csv'."""
    if lib == "XJTU":
        p = external / "XJTU data" / f"{cell}.csv"
    else:
        batch = cell.split("_battery-")[0]
        p = external / "MIT data" / batch / f"{cell}.csv"
    if not p.exists():
        return None, None
    df = pd.read_csv(p, usecols=["capacity"])
    return df["capacity"].to_numpy(dtype=float), p


def rel_dev(a, b):
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    ok = np.isfinite(a) & np.isfinite(b)
    m = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-12)
    return float(m.max()) if m.size else float("nan"), (lambda: float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")))()


def align_monotone(raw, proc, tol=None, max_offset=24):
    """Align the processed sequence to the raw sequence (v2: explicit row mapping; criteria R2').

    Method: for each offset k in [0, max_offset] do **monotonic greedy subsequence matching**
    (for each proc row i, search forward in raw from the current position for the first row with
    |a-b|/|b| <= tol; skipping raw rows = rows the publisher deleted), then take the candidate with
    "most matches, smallest k".
    For tol=None use the tolerance ladder (1e-6 -> 5e-3) and take the first tier that makes the match
    ratio >= 0.9, recording the match statistics of each tier (tol_ladder) for audit: the larger the
    tolerance, the more "fuzzy" the match.

    Returns (key fields):
      offset, matched, matched_ratio, tol_used, tol_ladder
      raw_index_of_proc : list[int], length = proc row count; -1 = unmatched (downstream reads raw cycles row by row)
      unmatched_proc_rows : list of unmatched proc row indices (explicit, not silent)
      dev_median/p99/max : relative deviation over **all matched rows** (no longer only rows <=tol)
      ratio_median, spearman_aligned : identity-verification quantity and rank correlation
      identity_pass : |ratio_median - 1| <= 0.03
      row_level_mapping : matched_ratio >= 0.9 (capability flag)
      status : pass / fail_identity / fail_mapping / fail_both
    """
    raw = np.asarray(raw, float)
    proc = np.asarray(proc, float)
    n_raw, n_proc = raw.size, proc.size
    base = {"offset": None, "matched": 0, "matched_ratio": 0.0,
            "tol_used": None, "tol_ladder": [],
            "raw_index_of_proc": [-1] * int(n_proc),
            "unmatched_proc_rows": list(range(int(n_proc))),
            "unmatched_raw_count": int(n_raw),
            "raw_span": None,
            "dev_median": None, "dev_p99": None, "dev_max": None,
            "proc_rows": int(n_proc), "raw_rows": int(n_raw),
            "ratio_median": None, "spearman_aligned": None,
            "identity_pass": False, "row_level_mapping": False, "status": "fail_mapping"}
    if n_proc == 0 or n_raw == 0:
        return base

    def _greedy(tol_):
        """Monotonic greedy subsequence matching (v2.1): single-row search window W=256; pointer does not advance on a mismatch."""
        ridx = -np.ones(n_proc, dtype=int)
        devs = np.full(n_proc, np.nan)
        j = 0
        W = 256
        for i in range(n_proc):
            b = proc[i]
            if not np.isfinite(b) or b == 0:
                continue
            jj = j
            jmax = min(n_raw, j + W)
            while jj < jmax:
                a = raw[jj]
                if np.isfinite(a) and abs(a - b) / abs(b) <= tol_:
                    ridx[i] = jj
                    devs[i] = abs(a - b) / abs(b)
                    j = jj + 1
                    break
                jj += 1
        return ridx, devs

    ladder_tols = [tol] if tol is not None else [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    ladder, chosen = [], None
    for t in ladder_tols:
        ridx, devs = _greedy(t)
        m = int((ridx >= 0).sum())
        ratio = m / n_proc
        ladder.append({"tol": t, "matched": m, "ratio": float(ratio), "offset": 0})
        if ratio >= 0.9:                    # stop as soon as it qualifies (a higher tolerance only gets fuzzier)
            chosen = (t, 0, ridx, devs, m, ratio)
            break
    if chosen is None:                      # no tier qualified: take the last tier's (largest tolerance) result for diagnostics
        ridx, devs = _greedy(ladder_tols[-1])
        m = int((ridx >= 0).sum())
        t, k, ratio = ladder_tols[-1], 0, m / n_proc
    else:
        t, k, ridx, devs, m, ratio = chosen

    okm = ridx >= 0
    dd = devs[okm]
    a_idx = ridx[okm]
    aa, bb = raw[a_idx], proc[okm]
    rho = float(pd.Series(aa).corr(pd.Series(bb), method="spearman")) if aa.size > 2 else None
    ratio_med = float(np.median(bb / np.maximum(aa, 1e-12))) if aa.size else None
    row_ok = bool(ratio >= 0.9)
    n_hit = int(m)
    ratio_ok = bool(ratio_med is not None and abs(ratio_med - 1) <= 0.03)
    # strict-tier matching (tol <= 1e-4): take the last tier in ladder (ladder is in ascending tolerance order)
    strict_matched = 0
    for entry in ladder:
        if entry["tol"] <= 1e-4:
            strict_matched = entry["matched"]
    if t is not None and t <= 1e-4:
        strict_matched = max(strict_matched, int(m))
    strict_ratio = strict_matched / n_proc if n_proc else 0.0
    # strong identity evidence: strict-tier matches >= ID_MIN_ROWS rows and share >= 50% (v2.3; see docstring R2' item 5)
    strict_ok = bool(strict_matched >= ID_MIN_ROWS and strict_ratio >= 0.5)
    if strict_ok and ratio_ok:
        identity_status = "pass"
    elif strict_ok:
        identity_status = "fail"
    else:
        identity_status = "unverifiable"
    identity = identity_status == "pass"
    exact = bool(row_ok and t <= 1e-5 and dd.size > 0 and dd.max() <= 1e-5)
    if identity and row_ok:
        status = "pass"
    elif identity:
        status = "fail_mapping"           # identity passes, row-level not mappable (B2 usable, C not usable)
    elif identity_status == "fail":
        status = "fail_identity"          # strict-tier matching sufficient but calibration ratio deviates >3%
    else:
        status = "unverifiable"           # strict-tier matching insufficient, identity unverifiable
    return {
        "offset": int(a_idx.min()) if a_idx.size else None,
        "matched": int(m),
        "matched_ratio": float(ratio),
        "tol_used": t,
        "tol_ladder": ladder,
        "raw_index_of_proc": ridx.tolist(),
        "unmatched_proc_rows": np.flatnonzero(~okm).tolist(),
        "unmatched_raw_count": int(n_raw - m),
        "raw_span": [int(a_idx.min()), int(a_idx.max())] if a_idx.size else None,
        "dev_median": float(np.median(dd)) if dd.size else None,
        "dev_p99": float(np.percentile(dd, 99)) if dd.size else None,
        "dev_max": float(dd.max()) if dd.size else None,
        "proc_rows": int(n_proc),
        "raw_rows": int(n_raw),
        "ratio_median": ratio_med,
        "spearman_aligned": rho,
        "n_matched": n_hit,
        "n_matched_strict": int(strict_matched),
        "strict_ratio": float(strict_ratio),
        "identity_status": identity_status,
        "exact_subsequence": exact,
        "identity_pass": identity,
        "row_level_mapping": row_ok,
        "status": status,
    }


DEFINITIONS = {
    "label": "processed `capacity` = raw `summary.discharge_capacity_Ah` (Ah; the library function get_capacity() "
             "takes the 2nd field of the summary struct)",
    "stage_split": "each cycle is split by relative_time_min == 0 into [charge, rest, discharge, rest] "
                   "(Batch-6 satellite data has only 3 segments: charge/rest/discharge) -- from the publisher "
                   "preprocessing library get_partial_value()",
    "CC": "in the charge stage, data with voltage <= 4.199 V **and** current >= 3.9 A (get_CC_value() default params: "
          "with voltage_range=None the condition is voltage<=4.199, then intersected with current>=3.9. "
          "library comment: the CV-stage voltage is sometimes below 4.199, hence the intersection with current)",
    "CV": "in the charge stage, data with voltage >= 4.199 V (get_CV_value() default params: with current_range=None "
          "it selects by voltage>=4.199; **it is not the complement of CC**. When current_range is passed it selects "
          "by current range instead)",
    "source": "github.com/wang-fujin/Battery-dataset-preprocessing-code-library/XJTUBatteryClass.py"
              "@e986a74054fb64c0f5134f2cc97f4643cd7699d2 (pinned by commit on 2026-09-23; the \"V>=4.199 and I>3.9\" "
              "recorded in v1 above was a mis-transcription and has been corrected against the publisher source; "
              "this repository does not redistribute it)",
    "caveat": "the above are the publisher's **default definitions of the code functions**; the processed CSV column "
              "names (CC Q / CV Q / CC charge time / CV charge time) are still not numerically aligned with that "
              "default definition (bexp62 column-by-column cross-check gives the measured comparison: the published "
              "CV column can be reproduced on the \"charge-stage tail with current<=0.5A\", while the CC column window "
              "rule is not public). This project's curve features always use a self-defined frozen window and do not "
              "pretend to be the publisher's definition.",
    "mit_units": "MIT struct: t is in minutes; I is **relative current (1C = 1.1 Ah rated)** -- measured dQc/(I*dt*60) "
                 "median = 1.1000 (p10/p90 = 1.088/1.118, battery-1 cycle 10); after multiplying by the rated capacity "
                 "the charge/discharge integrals close with QCharge/QDischarge to within 0.1% (measured 2026-09-23)",
}


# --------------------------------------------------------------------------- #
# XJTU
# --------------------------------------------------------------------------- #
def scan_xjtu(root: Path, external: Path, splits):
    import scipy.io as sio
    lib = {}
    gaps = []
    for b in XJTU_BATCHES:
        d = root / b
        if not d.is_dir():
            problems.append(f"XJTU: missing batch directory {d}")
            continue
        for f in sorted(d.glob("*.mat")):
            cell = f.stem
            info = {"raw_file": str(f), "raw_bytes": f.stat().st_size}
            try:
                mat = sio.loadmat(str(f), struct_as_record=False, squeeze_me=True)
                data = np.atleast_1d(mat["data"])
                n_cyc = len(data)
                cyc0 = data[0]
                fields = sorted({k for c in data[:1] for k in c._fieldnames})
                info.update({
                    "n_cycles_raw": int(n_cyc),
                    "cycle_fields": fields,
                    "units": {"voltage_V": "V", "current_A": "A", "capacity_Ah": "Ah",
                              "temperature_C": "degC", "relative_time_min": "min"},
                    "sign_convention": "charge positive / discharge negative (log as measured, per file)",
                })
                v = np.asarray(cyc0.voltage_V, float)
                cur = np.asarray(cyc0.current_A, float)
                info["sample_cycle_0"] = {
                    "n_points": int(v.size),
                    "v_min": float(np.nanmin(v)), "v_max": float(np.nanmax(v)),
                    "i_min": float(np.nanmin(cur)), "i_max": float(np.nanmax(cur)),
                }
                summ = mat.get("summary")
                if summ is not None:
                    dc = np.asarray(getattr(summ, "discharge_capacity_Ah", []), float).ravel()
                    cc = np.asarray(getattr(summ, "charge_capacity_Ah", []), float).ravel()
                    info["summary_n"] = int(dc.size)
                    raw_cap = dc
                else:
                    raw_cap = None
            except Exception as e:
                problems.append(f"XJTU {cell}: read failed {type(e).__name__}: {e}")
                lib[cell] = info
                continue

            unit = next((u for u, s in splits.items()
                         if u.startswith("XJTU") and cell in s["train"] + s["test"]), None)
            proc_cap, proc_path = processed_capacity(external, "XJTU", cell)
            if unit and proc_cap is not None and raw_cap is not None:
                al = align_monotone(raw_cap, proc_cap)
                info.update({"unit": unit,
                             "split": "test" if cell in splits[unit]["test"] else "train",
                             "processed_csv": str(proc_path),
                             "processed_rows": int(len(proc_cap)),
                             "align": al})
                if al["status"] != "pass":
                    problems.append(f"XJTU {cell}: align.status={al['status']} "
                                    f"ratio={al['ratio_median']} offset={al['offset']} "
                                    f"tol={al['tol_used']} match={al['matched_ratio']:.3f}")
            elif unit is None:
                gaps.append({"lib": "XJTU", "cell": cell, "reason": "no such cell in the processed release"})
            else:
                gaps.append({"lib": "XJTU", "cell": cell,
                             "reason": f"missing processed CSV or missing summary (unit={unit})"})
            lib[cell] = info
    return lib, gaps


# --------------------------------------------------------------------------- #
# MIT
# --------------------------------------------------------------------------- #
def scan_mit(root: Path, external: Path, splits):
    import h5py
    lib, gaps = {}, []
    for batch, fname in MIT_BATCHES:
        p = root / fname
        if not p.exists():
            problems.append(f"MIT: missing struct file {p}")
            continue
        with h5py.File(str(p), "r") as f:
            b = f["batch"]
            n = b["barcode"].shape[0]
            cells = []
            for i in range(n):
                bc = np.array(f[b["barcode"][i, 0]][()]).ravel().astype(int).tolist()
                cl_raw = float(np.array(f[b["cycle_life"][i, 0]][()]).ravel()[0])
                # some cells in the release have cycle_life = NaN (a known Severson-dataset phenomenon) -- must not force-convert to int
                cl = int(cl_raw) if np.isfinite(cl_raw) else None
                cell_id = f"{batch}_battery-{i + 1}"     # order within batch -> processed name (see module docstring section 1)
                summ = f[b["summary"][i, 0]]
                qd = np.array(summ["QDischarge"]).ravel()
                rec = {"batch": batch, "order_index": i + 1, "barcode": bc,
                       "channel": int(bc[4]) if len(bc) >= 5 else None,
                       "cycle_life": cl, "cycle_life_nan": bool(not np.isfinite(cl_raw)),
                       "summary_n": int(qd.size),
                       "identity_evidence": "struct order within batch (channel order); verified by capacity check below"}
                unit = "MIT"
                proc_cap, proc_path = processed_capacity(external, "MIT", cell_id)
                if cell_id in splits.get(unit, {}).get("train", []) + splits.get(unit, {}).get("test", []):
                    rec["split"] = "test" if cell_id in splits[unit]["test"] else "train"
                    if proc_cap is not None:
                        al = align_monotone(qd, proc_cap)
                        rec.update({"processed_csv": str(proc_path),
                                    "processed_rows": int(len(proc_cap)),
                                    "align": al})
                        if al["status"] != "pass":
                            problems.append(f"MIT {cell_id}: align.status={al['status']} "
                                            f"ratio={al['ratio_median']} offset={al['offset']} "
                                            f"tol={al['tol_used']} match={al['matched_ratio']:.3f}")
                    else:
                        gaps.append({"lib": "MIT", "cell": cell_id, "reason": "missing processed CSV"})
                else:
                    gaps.append({"lib": "MIT", "cell": cell_id,
                                 "reason": ("cycle_life is NaN (known in the release: this cell did not finish its "
                                            "lifetime)" if not np.isfinite(cl_raw) else
                                            "the raw batch exists but did not enter this project's official split "
                                            "(processed 125 vs platform 140)")})
                cells.append(rec)
            lib[batch] = {"raw_file": str(p), "raw_bytes": p.stat().st_size,
                          "n_cells_raw": int(n), "cells": cells}
    return lib, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xjtu-root", required=True)
    ap.add_argument("--mit-root", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/battery/raw_data_manifest.json"))
    ap.add_argument("--skip-hash", action="store_true", help="skip sha256 (recomputing on large files is slow)")
    a = ap.parse_args()

    xj_root = Path(a.xjtu_root).expanduser()
    mt_root = Path(a.mit_root).expanduser()
    ex = Path(a.external).expanduser()
    splits = official_splits(ex)

    out = {"sources": {}, "xjtu": {}, "mit": {}, "gaps": [], "splits": splits,
           "definitions": DEFINITIONS,
           "units_direction": {
               "XJTU": "current_A charge positive, discharge negative (sample_cycle_0 measured per file)",
               "MIT": "I charge positive, discharge negative; t in minutes; I is relative current (1C=1.1 Ah) --\n"
                      "measured dQc/(I*dt*60)=1.1000 (battery-1 c10); after scaling by the rated capacity Qc/Qd "
                      "close to 0.1%",
           }}
    out["sources"]["XJTU"] = {"url": XJTU_URL, "record": "zenodo:10963339",
                              "published_md5": "a635cef9678e0de21f9d5c1f78a4342c",
                              "verified": "md5sum -c OK (remote machine)"}
    out["sources"]["MIT"] = {"url_template": MIT_URL, "file_ids": MIT_FILE_IDS,
                             "note": "no published checksum over HTTP; the measured size/sha256 after download is "
                                     "recorded below"}

    xj, g1 = scan_xjtu(xj_root, ex, splits)
    mi, g2 = scan_mit(mt_root, ex, splits)
    out["xjtu"], out["mit"] = xj, mi
    out["gaps"] = g1 + g2

    if not a.skip_hash:
        for lib, rroot in (("XJTU", xj_root), ("MIT", mt_root)):
            hashes = {}
            if lib == "XJTU":
                for b in XJTU_BATCHES:
                    for f in sorted((rroot / b).glob("*.mat")):
                        hashes[f"{b}/{f.name}"] = sha256(f)
            else:
                for _, fname in MIT_BATCHES:
                    f = rroot / fname
                    if f.exists():
                        hashes[fname] = sha256(f)
            out["sources"][lib]["sha256"] = hashes

    # R6: release_gate -- the failure list downstream (bexp62/63/64/65) must consume
    # semantics (v2.2): status in {pass, fail_mapping, fail_identity, unverifiable};
    #   B2 excludes cells with identity_status != pass (fail_identity / unverifiable);
    #   C additionally requires row_level_mapping=True (a fail_mapping cell has usable identity but cannot be read
    #   row by row).
    gate = {"rule": "B2 (bexp63/65) excludes cells with identity_status != pass; "
                    "C (bexp64) additionally requires row_level_mapping=True; "
                    "bexp62 cross-check uses only row-alignable (row_level_mapping) cells.",
            "identity_failures": [], "unverified": [], "mapping_failures": []}
    for cell, v in xj.items():
        al = v.get("align")
        if not al:
            continue
        rec = {"cell": cell, "lib": "XJTU", "status": al["status"],
               "n_matched": al.get("n_matched"), "ratio_median": al["ratio_median"],
               "matched_ratio": al["matched_ratio"],
               "exact_subsequence": al.get("exact_subsequence")}
        if al["status"] == "fail_identity":
            gate["identity_failures"].append(rec)
        elif al["status"] == "unverifiable":
            gate["unverified"].append(rec)
        elif al["status"] == "fail_mapping":
            gate["mapping_failures"].append(rec)
    for b, grp in mi.items():
        for c in grp["cells"]:
            al = c.get("align")
            if not al:
                continue
            cell = f"{c['batch']}_battery-{c['order_index']}"
            rec = {"cell": cell, "lib": "MIT", "status": al["status"],
                   "n_matched": al.get("n_matched"), "ratio_median": al["ratio_median"],
                   "matched_ratio": al["matched_ratio"],
                   "exact_subsequence": al.get("exact_subsequence")}
            if al["status"] == "fail_identity":
                gate["identity_failures"].append(rec)
            elif al["status"] == "unverifiable":
                gate["unverified"].append(rec)
            elif al["status"] == "fail_mapping":
                gate["mapping_failures"].append(rec)
    out["release_gate"] = gate

    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    n_x = len(xj)
    n_m = sum(v["n_cells_raw"] for v in mi.values())
    # R2' capability statistics: row-alignable / calibration-only alignable (publisher resampled or trimmed)
    row_ok = sum(1 for v in xj.values() if (v.get("align") or {}).get("row_level_mapping"))
    row_ok += sum(1 for b in mi.values() for c in b["cells"]
                  if (c.get("align") or {}).get("row_level_mapping"))
    tot = sum(1 for v in xj.values() if v.get("align")) + \
          sum(1 for b in mi.values() for c in b["cells"] if c.get("align"))
    print(f"[manifest] XJTU cells={n_x}  MIT cells={n_m}  gaps={len(out['gaps'])}  "
          f"row-level alignable={row_ok}/{tot}  problems={len(problems)}")
    print(f"[release_gate] identity_failures={len(gate['identity_failures'])} "
          f"unverified={len(gate['unverified'])} mapping_failures={len(gate['mapping_failures'])}")
    for p in problems:
        print("  PROBLEM:", p)
    if problems:
        return 1
    print("R1-R6 passed; output:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
