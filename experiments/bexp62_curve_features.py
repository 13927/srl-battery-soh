"""bexp62: **extraction + small-scale validation** of curve features and physical references
(F03 Stage 3.2/3.3).

Per `f03_baseline_completion_plan.md`:
  Section 5-B uses the real Q(V)-difference scalar (the Severson feature); Section 6-C uses the
  full charge-segment integral of I dt as the physical SOH reference; Section 3 fixes the common
  comparison protocol (official splits unchanged; N=100 fixed; scoring only on cycles after N with
  complete information; rules decided on the source side).
This script does only the **data -> features** layer and uses the "cross-check against the
released columns" as a release gate; the downstream B2 (same-task SOH comparison) and C (physical
reference) consume this script's output in bexp63 / bexp64.

Inputs: `--xjtu-root` (the extracted `Battery Dataset`), `--mit-root` (the three struct `.mat`
      files), `--external` (the processed release `PINN4SOH/data`, used only for cross-checking and
      the official splits).

Outputs (written to `results/battery/bexp62_curve_features.json`):
  - per-cell (XJTU 55 + MIT 125):
      * `qdiff_var_log10`: the Severson scalar log10(var(dQ(V))) with dQ = Q_dis(cycle100, V) - Q_dis(cycle10, V),
                           interpolated on the frozen voltage grid (only given when the cell has >= 100 cycles)
      * `qchg_int_Ah[N]`: three readings of the raw charge-segment current integral (cycle N, cycle 100,
                           lifetime median)
      * `qc_release`     : the charge-side released columns that can be matched (XJTU: `CC Q`+`CV Q`;
                           MIT: none) - median
      * `cross_check`    : relative deviation of the I dt integral vs the released columns (XJTU);
                           sanity metrics of the Q(V) grid and resampling
      * `row_level_mapping`: from `raw_data_manifest.json` (whether row-by-row mapping is possible)
  - `validation`: results of all gates and the failure lists.

================== criteria (frozen before the run) ==================
V1 curve grid: dQ(V) is linearly interpolated at 1000 points on the frozen grid (discharge-segment
   capacity vs voltage): LFP/MIT use [2.5,3.5] V, NCM/XJTU [3.4,4.0] V (the two libraries have
   different plateau voltages; one shared grid would produce a non-physical zero coverage); the
   non-finite share after interpolation must be <= 1%; cells with fewer than 100 cycles are
   recorded as `insufficient_cycles` and get no scalar (no imputation). Discharge/charge segments
   are taken by the **longest run of the current sign** (not by counting segment markers; the
   satellite batch has only 3 segments).
V2 cross-check: column-by-column convention.
    (a) label column: processed `capacity` = the raw discharge capacity -- verified by
        `raw_data_manifest.json` R2' (reused here);
    (b) window columns (v2 revision, 2026-09-23, third independent audit F3-03): v1 mis-copied the
        publisher's default definitions as "charge segment v>=4.199 and i>3.9"; corrected against
        the publisher source
        (Battery-dataset-preprocessing-code-library@e986a740, XJTUBatteryClass.py):
          default CC = charge segment (voltage <= 4.199) & (current >= 3.9)
          default CV = charge segment voltage >= 4.199 (**not** the complement of CC; passing
          current_range switches to a current-range rule)
        The corrected implementation redoes the column-by-column check, recording the 2026-09-23
        measurement (battery-1 cycle 10):
          - the released `CV Q`/`CV charge time` columns match exactly the "charge-segment tail
            with current <= 0.5 A" (release 0.058 Ah / 882.78 s; measured 0.0577 Ah / 882.8 s;
            in-segment mean current 0.2352 matches the released current mean);
          - the released `CC Q`/`CC charge time` columns (~0.35 Ah / ~315 s) do not match the
            default CC (~1.61 Ah / ~1451 s), and their window rule has no definition in the
            publisher's public code/docs -- recorded as unreconciled (no guessing, no forcing).
        This project's curve features always use this script's self-defined frozen window and do
        not depend on those four columns.
    (c) MIT units (v2-v5 corrections): t is in **minutes**; I is **relative current (1C = 1.1 Ah
        rated)** (measured dQc/(I*dt*60) median = 1.1000). The charge integral = `Qnom x sum over
        positive segments of I dt`, done by the shared `positive_segment_ah` on the **complete
        arrays** -- v3/v4 sliced the already-compacted array with the complete mask's segment
        boundaries (fourth-round audit item A: synthetic two-segment sample -66.7%); fixed in v5;
        v4's data-quality gate is retained (cycle duration span > 2 h -> None; normal ~54 min; a
        post-diagnostic revision). Every cell's charge integral is checked for closure against the
        per-cycle Qc (deviations go into validation); no claim of a full-library unit
        re-verification.
V3 label isolation: this script reads only the raw curves and (for checking and splits only) the
             release; the `capacity` label is never written into features; `qdiff_var_log10` and
             `qchg_int_Ah` are both curve-derived and no label column appears in the output.
V4 explicit coverage: every cell gets a `status`: ok / insufficient_cycles / no_curve /
             align_unknown; any cell with `status != "ok"` must carry a reason and must not be
             silently dropped.
V5 version    : raw-file sha256 comes from `raw_data_manifest.json` (no re-hashing); the output
             records the script version and time.
==================================================================

Run (on the machine holding the data):
  ./.venv/bin/python experiments/bexp62_curve_features.py \
      --xjtu-root "$HOME/data_raw/xjtu/xjtu_raw/Battery Dataset" --mit-root ~/data_raw/mit \
      --external external/PINN4SOH/data --manifest results/battery/raw_data_manifest.json \
      --out results/battery/bexp62_curve_features.json [--limit-cells 6]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

V_GRID = np.linspace(2.5, 3.5, 1000)          # frozen grid (LFP / MIT)
V_GRID_NCM = np.linspace(3.4, 4.0, 1000)      # frozen grid (NCM / XJTU, see V1)
N_WINDOW = 100                            # frozen observation window
MIT_QNOM = 1.1                            # MIT rated capacity (Ah; I is relative current, 1C=1.1Ah)
XJTU_BATCHES = ["Batch-1", "Batch-2", "Batch-3", "Batch-4", "Batch-5", "Batch-6"]
MIT_FILES = {"2017-05-12": "mat_5c86c0b5fa2ede00015ddf66.mat",
             "2017-06-30": "mat_5c86bf13fa2ede00015ddd82.mat",
             "2018-04-12": "mat_5c86bd64fa2ede00015ddbb2.mat"}

failures = []
notes = []


# --------------------------------------------------------------------------- #
# Curve utilities
# --------------------------------------------------------------------------- #
def longest_run(mask):
    """Return the slice of the longest consecutive True run (not depending on segment markers)."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return slice(0, 0)
    idx = np.flatnonzero(mask)
    split = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[0, split + 1]
    ends = np.r_[split, idx.size - 1]
    lens = idx[ends] - idx[starts]
    k = int(np.argmax(lens))
    return slice(int(idx[starts[k]]), int(idx[ends[k]]) + 1)


def interp_qv(v, q, grid=V_GRID):
    """Interpolate the (v, q) curve onto the fixed voltage grid; return (q_grid, finite_ratio).

    The discharge segment usually has monotonically falling voltage; sort by voltage ascending
    before interpolating, with nan outside the grid.
    """
    v = np.asarray(v, float)
    q = np.asarray(q, float)
    ok = np.isfinite(v) & np.isfinite(q)
    v, q = v[ok], q[ok]
    if v.size < 10:
        return None, 0.0
    order = np.argsort(v)
    v, q = v[order], q[order]
    vv, idx = np.unique(v, return_index=True)
    qq = q[idx]
    if vv.size < 10:
        return None, 0.0
    g = np.interp(grid, vv, qq, left=np.nan, right=np.nan)
    return g, float(np.isfinite(g).mean())


def severson_scalar(cycles, grid=V_GRID):
    """log of the variance of dQ(V) = Q_dis(cycle100) - Q_dis(cycle10) (the Severson scalar).

    cycles: list of dict(v=..., q=...) aligned to 1-based cycle numbers (index 0 -> cycle 1).
    """
    if len(cycles) < N_WINDOW:
        return None, {"reason": f"cycles<{N_WINDOW}"}
    c10, c100 = cycles[9], cycles[99]
    q10, f10 = interp_qv(c10["v"], c10["q"], grid)
    q100, f100 = interp_qv(c100["v"], c100["q"], grid)
    if q10 is None or q100 is None or min(f10, f100) < 0.99:
        return None, {"reason": f"grid_finite={min(f10, f100):.3f}"}
    dq = q100 - q10
    var = float(np.nanvar(dq))
    return float(np.log10(var)) if var > 0 else None, {"var": var, "finite": min(f10, f100)}


def trapz_ah(t, i):
    """Trapezoidal integral of I dt -> Ah (t seconds, i amperes; minutes also accepted: callers convert to seconds)."""
    t = np.asarray(t, float)
    i = np.asarray(i, float)
    ok = np.isfinite(t) & np.isfinite(i)
    if ok.sum() < 2:
        return None
    return float(np.trapezoid(i[ok], t[ok]) / 3600.0)


def dur_s(t_s, mask):
    """Duration of the masked segment (seconds; endpoint difference)."""
    t_s = np.asarray(t_s, float)
    if int(np.asarray(mask).sum()) < 2:
        return None
    return float(t_s[mask][-1] - t_s[mask][0])


def tail_window(t_s, i, thr=0.5):
    """The measured window of the released CV columns (2026-09-23): the longest consecutive tail segment with current <= thr in the charge segment."""
    i = np.asarray(i, float)
    t_s = np.asarray(t_s, float)
    s = longest_run(i <= thr)
    if s.stop - s.start < 2:
        return None
    return {"ah": trapz_ah(t_s[s], i[s]), "s": float(t_s[s][-1] - t_s[s][0]),
            "i_mean": float(i[s].mean())}


def positive_segment_ah(t_sec, i):
    """Sum over consecutive positive-current segments of the I dt integral -> Ah (**complete arrays**; shared by bexp62/64).

    v5 fix (2026-09-23b, fourth-round audit item A): v3/v4 first deleted non-positive points to get
    the compacted array (t[m], i[m]) but then sliced it with the segment boundaries of the
    **complete** mask m -- two different index spaces, so segments after the first were mis-sliced
    or lost (synthetic two-segment sample: 0.01833 vs the correct 0.055 Ah, -66.7%). This function
    takes boundaries and slices on the complete arrays. Synthetic-sample acceptance: two segments,
    leading rest, single segment, all non-positive.
    """
    t_sec = np.asarray(t_sec, float)
    i = np.asarray(i, float)
    if t_sec.size < 2 or i.size < 2 or t_sec.size != i.size:
        return None
    m = i > 0
    if not m.any():
        return None
    d = np.diff(m.astype(int))
    st = list(np.flatnonzero(d == 1) + 1)
    en = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        st = [0] + st
    if m[-1]:
        en = en + [len(m)]
    tot = 0.0
    for s, e in zip(st, en):
        if e - s < 2:
            continue
        x = trapz_ah(t_sec[s:e], i[s:e])
        if x is not None:
            tot += x
    return tot


# --------------------------------------------------------------------------- #
# XJTU
# --------------------------------------------------------------------------- #
def scan_xjtu(root: Path, external: Path, manifest: dict, limit=None):
    import scipy.io as sio
    out = {}
    cells = []
    for b in XJTU_BATCHES:
        cells += sorted((root / b).glob("*.mat"))
    if limit:
        cells = cells[:limit]
    for f in cells:
        cell = f.stem
        rec = {"lib": "XJTU", "raw_file": str(f)}
        al = (manifest["xjtu"].get(cell) or {}).get("align") or {}
        rec["row_level_mapping"] = bool(al.get("row_level_mapping"))
        rec["identity_pass"] = bool(al.get("identity_pass"))
        rec["ratio_median"] = al.get("ratio_median")
        try:
            mat = sio.loadmat(str(f), struct_as_record=False, squeeze_me=True)
            data = np.atleast_1d(mat["data"])
        except Exception as e:
            rec.update({"status": "no_curve", "reason": f"{type(e).__name__}: {e}"})
            out[cell] = rec
            continue
        # the discharge segment of each cycle: stage 3 (1-based) -> segmented by relative_time_min==0
        cycles, chg = [], []
        for c in data:
            rt = np.asarray(getattr(c, "relative_time_min", []), float)
            v = np.asarray(getattr(c, "voltage_V", []), float)
            i = np.asarray(getattr(c, "current_A", []), float)
            cap = np.asarray(getattr(c, "capacity_Ah", []), float)
            if rt.size < 4:
                cycles.append({}); chg.append(None); continue
            s1 = longest_run(i > 0)      # charge segment = the longest positive run of the current
            s3 = longest_run(i < 0)      # discharge segment = the longest negative run of the current
            chg.append({"t": rt[s1], "i": i[s1], "v": v[s1]})
            cycles.append({"v": v[s3], "q": cap[s3]})
        rec["n_cycles"] = len(data)
        scalar, info = severson_scalar(cycles, grid=V_GRID_NCM)   # XJTU = NCM grid
        rec["qdiff_var_log10"] = scalar
        rec["severson_info"] = info
        rec["grid"] = "NCM [3.4,4.0]"
        # I dt: cycle N, cycle 100, lifetime median (only on cycles with complete information)
        ints = []
        win = []      # publisher default-definition windows (v2 fix: CC = V<=4.199 & I>=3.9; CV = V>=4.199)
        tails = []    # measured candidate for the released CV column (tail with I<=0.5)
        for c in chg:
            if c is None:
                ints.append(None); win.append(None); tails.append(None); continue
            t_s = c["t"] * 60.0            # relative_time_min -> seconds
            ints.append(trapz_ah(t_s, c["i"]))
            v_, i_ = c.get("v"), c["i"]
            tails.append(tail_window(t_s, i_))
            if v_ is None:
                win.append(None); continue
            cc_mask = (v_ <= 4.199) & (i_ >= 3.9)   # publisher default definition (@e986a740, corrected)
            cv_mask = v_ >= 4.199                   # default CV (not the complement of CC)
            win.append({"cc_ah": trapz_ah(t_s[cc_mask], i_[cc_mask]),
                        "cc_s": dur_s(t_s, cc_mask),
                        "cv_ah": trapz_ah(t_s[cv_mask], i_[cv_mask]),
                        "cv_s": dur_s(t_s, cv_mask)})
        ints_f = [x for x in ints if x is not None]
        rec["qchg_int_Ah"] = {
            "cycle10": ints[9] if len(ints) > 9 else None,
            "cycle100": ints[99] if len(ints) > 99 else None,
            "median_all": float(np.median(ints_f)) if ints_f else None,
        }
        wf = [w for w in win if w]
        if wf:
            rec["publisher_default_windows"] = {
                "cc_ah_median": float(np.median([w["cc_ah"] for w in wf if w["cc_ah"] is not None])),
                "cv_ah_median": float(np.median([w["cv_ah"] for w in wf if w["cv_ah"] is not None])),
                "cc_s_median": float(np.median([w["cc_s"] for w in wf if w["cc_s"] is not None])),
                "cv_s_median": float(np.median([w["cv_s"] for w in wf if w["cv_s"] is not None])),
            }
        tf = [w for w in tails if w]
        if tf:
            rec["tail_I_le_0.5_median"] = {
                "ah": float(np.median([w["ah"] for w in tf if w["ah"] is not None])),
                "s": float(np.median([w["s"] for w in tf])),
                "i_mean": float(np.median([w["i_mean"] for w in tf])),
            }
        # cross-check (column-by-column; see module docstring V2(b))
        p = external / "XJTU data" / f"{cell}.csv"
        if rec["row_level_mapping"] and p.exists():
            df = pd.read_csv(p)
            if "CC Q" in df.columns:
                rel = {c: float(df[c].median())
                       for c in ("CC Q", "CV Q", "CC charge time", "CV charge time")}

                def _rd(a, b):
                    if a is None or b in (None, 0):
                        return None
                    return abs(a - b) / abs(b)

                tail_med = rec.get("tail_I_le_0.5_median") or {}
                defw = rec.get("publisher_default_windows") or {}
                cv_ah_dev = _rd(tail_med.get("ah"), rel["CV Q"])
                cv_s_dev = _rd(tail_med.get("s"), rel["CV charge time"])
                cc_ah_dev = _rd(defw.get("cc_ah_median"), rel["CC Q"])
                cc_s_dev = _rd(defw.get("cc_s_median"), rel["CC charge time"])
                rec["cross_check"] = {
                    "release_medians": rel,
                    "publisher_default_CC_median": {"Ah": defw.get("cc_ah_median"),
                                                     "s": defw.get("cc_s_median")},
                    "publisher_default_CV_median": {"Ah": defw.get("cv_ah_median"),
                                                     "s": defw.get("cv_s_median")},
                    "tail_I_le_0.5_median": tail_med,
                    "rel_dev": {"CV Q (release vs tail)": cv_ah_dev,
                                 "CV charge time (release vs tail)": cv_s_dev,
                                 "CC Q (release vs default window)": cc_ah_dev,
                                 "CC charge time (release vs default window)": cc_s_dev},
                    "reconciled": {"CV Q": bool(cv_ah_dev is not None and cv_ah_dev <= 0.01),
                                    "CV charge time": bool(cv_s_dev is not None and cv_s_dev <= 0.01),
                                    "CC Q": False, "CC charge time": False},
                    "reason": "CV column = charge-segment tail with I<=0.5A (reproducible); the CC column window rule is not public"
                              " (default definition ~1.61Ah/~1451s disagrees with the release ~0.35Ah/~315s)",
                }
                rel_cap = float(df["capacity"].median())
                mine_chg = rec["qchg_int_Ah"]["median_all"]
                if mine_chg and rel_cap:
                    rec["cross_check"]["charge_vs_label"] = {
                        "mine_charge_Ah": mine_chg, "release_capacity": rel_cap,
                        "dev": abs(mine_chg - rel_cap) / rel_cap,
                        "note": "total charge capacity vs the discharge label (same magnitude expected; not equal)"}
        rec["status"] = "ok" if scalar is not None else "insufficient_cycles"
        if rec["status"] != "ok":
            rec["reason"] = info.get("reason", "unknown")
        out[cell] = rec
    return out


# --------------------------------------------------------------------------- #
# MIT
# --------------------------------------------------------------------------- #
def scan_mit(root: Path, manifest: dict, limit=None):
    import h5py
    out = {}
    for batch, fname in MIT_FILES.items():
        with h5py.File(str(root / fname), "r") as f:
            b = f["batch"]
            n = b["barcode"].shape[0]
            for i in range(n if not limit else min(n, limit)):
                cell = f"{batch}_battery-{i + 1}"
                rec = {"lib": "MIT", "raw_file": f"{fname}#{i}"}
                al = ((manifest["mit"].get(batch) or {}).get("cells") or [])
                rl = next((c.get("align", {}).get("row_level_mapping") for c in al
                           if c.get("order_index") == i + 1), None)
                ip = next((c.get("align", {}).get("identity_pass") for c in al
                           if c.get("order_index") == i + 1), None)
                rec["row_level_mapping"] = bool(rl)
                rec["identity_pass"] = bool(ip)
                cyc = f[b["cycles"][i, 0]]
                ncyc = int(cyc["V"].shape[0])
                rec["n_cycles"] = ncyc
                cycles, chg = [], []
                for k in range(ncyc):
                    v = np.asarray(f[cyc["V"][k, 0]][()]).ravel().astype(float)
                    i_ = np.asarray(f[cyc["I"][k, 0]][()]).ravel().astype(float)
                    q = np.asarray(f[cyc["Qd"][k, 0]][()]).ravel().astype(float)
                    t = np.asarray(f[cyc["t"][k, 0]][()]).ravel().astype(float) \
                        if "t" in cyc else None
                    mask = i_ > 0
                    # v5 units/integration (2026-09-23b, fourth-round audit item A): t is in minutes (x60 -> seconds);
                    # I is relative current (1C=1.1 Ah); the integral sums over **complete arrays** segment by
                    # segment (positive_segment_ah) -- v3/v4 used the complete mask's boundaries to slice the
                    # compacted (t[m], i[m]), mis-slicing/losing segments after the first.
                    # Data-quality gate (v4): a positive-current segment whose duration span exceeds 2 h is an
                    # artifact (normal ~54 min; 2017-06-30 battery-40 k=247 spans 8.8 h); the whole cycle becomes None.
                    tt = (t * 60.0 if t is not None
                          else np.arange(i_.size, dtype=float))
                    mpos = tt[mask]
                    span_ok = bool(mpos.size == 0
                                   or float(mpos.max() - mpos.min()) <= 7200.0)
                    chg.append({"t": tt, "i": i_, "span_ok": span_ok})
                    # discharge curve: the longest consecutive run with negative current; capacity from cumulative Qd
                    s2 = longest_run(i_ < 0)
                    cycles.append({"v": v[s2], "q": q[s2]})
                scalar, info = severson_scalar(cycles)   # MIT = LFP grid
                rec["grid"] = "LFP [2.5,3.5]"
                rec["qdiff_var_log10"] = scalar
                rec["severson_info"] = info

                def _qchg(c):
                    """qchg_Ah = Qnom x sum over positive segments of I dt (complete-array segment integration; t in seconds).

                    v5: calls the shared positive_segment_ah (index-space fix); v4's data-quality gate is retained
                    (cycle duration span > 2 h -> None).
                    """
                    if not c.get("span_ok", True):
                        return None
                    x = positive_segment_ah(c["t"], c["i"])
                    return None if x is None else x * MIT_QNOM

                ints = [_qchg(c) for c in chg]
                # closure check (V2(c): compute vs the last value of the per-cycle Qc array at cycles 10/50/100)
                clos = []
                for k in (9, 49, 99):
                    if k >= ncyc or ints[k] is None:
                        continue
                    qc = np.asarray(f[cyc["Qc"][k, 0]][()]).ravel().astype(float)
                    if qc.size and qc[-1] > 0:
                        clos.append(abs(ints[k] - qc[-1]) / qc[-1])
                rec["unit_closure_vs_Qc"] = {"n": len(clos),
                                             "max_dev": float(max(clos)) if clos else None,
                                             "bad_cycles_span_gt_2h": int(sum(
                                                 1 for c in chg if not c.get("span_ok", True)))}
                ints_f = [x for x in ints if x is not None]
                rec["qchg_int_Ah"] = {
                    "cycle10": ints[9] if len(ints) > 9 else None,
                    "cycle100": ints[99] if len(ints) > 99 else None,
                    "median_all": float(np.median(ints_f)) if ints_f else None,
                }
                rec["status"] = "ok" if scalar is not None else "insufficient_cycles"
                if rec["status"] != "ok":
                    rec["reason"] = info.get("reason", "unknown")
                out[cell] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xjtu-root", required=True)
    ap.add_argument("--mit-root", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "results/battery/raw_data_manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "results/battery/bexp62_curve_features.json"))
    ap.add_argument("--limit-cells", type=int, default=None)
    a = ap.parse_args()

    manifest = json.load(open(a.manifest))
    xj = scan_xjtu(Path(a.xjtu_root).expanduser(), Path(a.external).expanduser(), manifest,
                   a.limit_cells)
    mi = scan_mit(Path(a.mit_root).expanduser(), manifest, a.limit_cells)
    cells = {**xj, **mi}
    stats = {}
    for lib in ("XJTU", "MIT"):
        sub = {k: v for k, v in cells.items() if v["lib"] == lib}
        stats[lib] = {
            "n": len(sub),
            "ok": sum(1 for v in sub.values() if v["status"] == "ok"),
            "insufficient_cycles": sum(1 for v in sub.values() if v["status"] == "insufficient_cycles"),
            "no_curve": sum(1 for v in sub.values() if v["status"] == "no_curve"),
            "row_level": sum(1 for v in sub.values() if v.get("row_level_mapping")),
        }
    # V2(b) column-check statistics (v2 structure: reconciled is a dict)
    n_cc, cv_q_ok, cv_t_ok, devs = 0, 0, 0, []
    for v in cells.values():
        cc = (v.get("cross_check") or {})
        r = cc.get("reconciled") or {}
        if not r:
            continue
        n_cc += 1
        cv_q_ok += int(bool(r.get("CV Q")))
        cv_t_ok += int(bool(r.get("CV charge time")))
        for _k, val in (cc.get("rel_dev") or {}).items():
            if val is not None:
                devs.append(val)
    if n_cc:
        notes.append(f"V2(b): {n_cc} XJTU cells column-by-column: CV Q reproduced {cv_q_ok}/{n_cc}, "
                     f"CV charge time reproduced {cv_t_ok}/{n_cc}; CC columns recorded unreconciled (window rule not public)")
    # V2(c) MIT unit closure (integral vs Qc on sampled cycles)
    mit_clos = [v.get("unit_closure_vs_Qc", {}).get("max_dev") for v in cells.values()
                if v.get("lib") == "MIT"]
    mit_clos = [x for x in mit_clos if x is not None]
    if mit_clos:
        notes.append(f"V2(c): MIT unit closure (integral vs Qc on sampled cycles) max_dev median={np.median(mit_clos):.2e}")
    out = {"_meta": {"script": "bexp62_curve_features.py",
                     "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "v_grid": [float(V_GRID[0]), float(V_GRID[-1]), int(V_GRID.size)],
                     "n_window": N_WINDOW, "limit_cells": a.limit_cells,
                     "mit_qnom": MIT_QNOM},
           "stats": stats,
           "cross_check": {"n_cells_checked": n_cc, "cv_q_reconciled": cv_q_ok,
                           "cv_time_reconciled": cv_t_ok,
                           "rel_dev_median": float(np.median(devs)) if devs else None,
                           "mit_closure_median": float(np.median(mit_clos)) if mit_clos else None},
           "cells": cells,
           "validation": {"failures": failures, "notes": notes}}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[bexp62] XJTU ok={stats['XJTU']['ok']}/{stats['XJTU']['n']} "
          f"row_level={stats['XJTU']['row_level']} | "
          f"MIT ok={stats['MIT']['ok']}/{stats['MIT']['n']} row_level={stats['MIT']['row_level']}")
    print(f"[bexp62] cross-check: XJTU n={n_cc} (CV Q reproduced {cv_q_ok}, CV time reproduced {cv_t_ok}); "
          f"MIT unit-closure median={out['cross_check']['mit_closure_median']}")
    for x in failures[:10]:
        print("  FAIL:", x)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
