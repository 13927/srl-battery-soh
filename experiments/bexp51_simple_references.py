"""bexp51: minimal physical references -- closing the last two items of Reviewer 1 (R1-M5)
("placing the ML numbers").

Motivation: the reviewer asked for a minimal physical/statistical reference that an
electrochemistry reader would accept, to place the ML numbers. This script is
deliberately kept simple: it does not aim for performance, only for "existing and
interpretable".

Reference (1)  Coulomb-counting style (no training, no learning at all)
  Per cell: pred_t = (CC Q + CV Q)_t / (CC Q + CV Q)_1, i.e. the self-ratio of the
  charge-capacity sequence estimates SOH; the per-cell RMSE is computed directly
  against the official SOH label y_t; aggregated per unit over the 11 units.
  * Key: it uses the **test cell's own cycle 1** -- which is exactly the paper's
    central point (self-reference) -- and there is no learning here. It is
    deliberately a minimal reference, not the strongest Coulomb-counting implementation.

Reference (2)  Severson-style early-cycle scalar (adapted, since the processed data
               has no voltage curves)
  The processed CSVs contain no voltage curves, so Severson 2019's
  log-variance-of-dQ(V) cannot be reproduced; it is replaced by an **early-cycle
  capacity-fade slope**:
    - for each cell take the first N cycles of capacity (here the SOH label y is used;
      it differs from capacity only by a per-unit constant, absorbed by the linear
      regression) and fit a line to obtain the slope k;
    - fit a linear model (k, first-cycle SOH) -> end-of-life SOH (y[-1]) on the
      **source cells**;
    - for a **held-out cell** use the prediction as a constant over the cell's whole
      SOH trajectory and compute the per-cell RMSE.
  N=100; if the shortest cell of the unit has fewer than 100 cycles use 50; if still
  shorter use the shortest length (the per-unit N is recorded as measured).

================== decision criteria (recorded in this docstring; prior registration is not claimed -- see Section 2.7) ==================
B1: each of the two reference baselines gives a **finite (non-divergent)** per-cell
    RMSE on >= 7/11 units, and each one's median error is **higher** than GBDT-SRL's
    median error (they are expected to be worse -- used to place the ML numbers).
    Any counter-example (a reference is better on most units, or diverges) must be
    reported as measured; it must not be dressed up.
========================================================================

Contrast: GBDT-raw uses the cell-macro mean of bexp14 (author_minmax, cap=2048, 5 seeds);
      GBDT-SRL uses the cell-macro median of bexp50 (same pipeline A). If either is
      missing the column shows "--".
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LinearRegression

from battery_lab.data_adapters import FEATURE_COLS, FIT_UNITS, load_fit_unit
from battery_lab.protocols import CONTEXT_CAP

OUT = ROOT / "results/battery/bexp51_simple_refs.jsonl"
B14 = ROOT / "results/battery/bexp14_gbdt.jsonl"
B50 = ROOT / "results/battery/bexp50_linear.jsonl"
CC_IDX = FEATURE_COLS.index("CC Q")
CV_IDX = FEATURE_COLS.index("CV Q")
N_PREF = [100, 50]           # prefer N=100, fall back to 50, then to the shortest length


# --------------------------------------------------------------- reference 1 -
def coulomb_ref(cells):
    """Coulomb-counting self-ratio -> per-cell RMSE; returns (macro, per_cell, finite_flags)."""
    per_cell, finite = {}, []
    for c in cells:
        q = c.X[:, CC_IDX] + c.X[:, CV_IDX]
        if q[0] == 0 or not np.isfinite(q[0]):
            per_cell[c.cell_id] = float("nan"); finite.append(False); continue
        pred = q / q[0]
        rmse = float(np.sqrt(np.mean((c.y - pred) ** 2)))
        per_cell[c.cell_id] = rmse
        finite.append(bool(np.isfinite(rmse)))
    vals = [v for v in per_cell.values() if np.isfinite(v)]
    macro = float(np.mean(vals)) if vals else float("nan")
    return macro, per_cell, all(finite)


# --------------------------------------------------------------- reference 2 -
def pick_N(cells):
    """Determine the early window N from the shortest cell of the unit."""
    minlen = min(len(c.y) for c in cells)
    for n in N_PREF:
        if minlen >= n:
            return n, False
    return minlen, True   # degraded: use the shortest length


def severson_ref(train_cells, test_cells, N):
    """Fit (k, first-cycle SOH) on source cells -> end-of-life SOH; held-out cells get a constant prediction over the whole trajectory."""
    def feats(c):
        y = c.y[:N]
        k = float(np.polyfit(np.arange(N), y, 1)[0])
        return [k, float(c.y[0])], float(c.y[-1])

    Xs, ts = [], []
    for c in train_cells:
        (f, t) = feats(c); Xs.append(f); ts.append(t)
    if len(Xs) < 2:
        return float("nan"), {}, float("nan")
    model = LinearRegression().fit(np.array(Xs), np.array(ts))

    per_cell, term_abs = {}, {}
    for c in test_cells:
        f, _ = feats(c)
        pred = float(model.predict(np.array([f]))[0])
        per_cell[c.cell_id] = float(np.sqrt(np.mean((c.y - pred) ** 2)))
        term_abs[c.cell_id] = float(abs(pred - c.y[-1]))
    macro = float(np.mean(list(per_cell.values()))) if per_cell else float("nan")
    term_macro = float(np.mean(list(term_abs.values()))) if term_abs else float("nan")
    return macro, per_cell, term_macro


# ------------------------------------------------------------ GBDT contrast --
def load_refs():
    gbdt_raw, gbdt_srl = {}, {}
    if B14.exists():
        rows = [json.loads(l) for l in B14.read_text().splitlines()]
        for uid, _, _ in FIT_UNITS:
            v = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
                 and r["view"] == "raw" and r["normalize"] == "author_minmax"
                 and r["cap"] == CONTEXT_CAP]
            gbdt_raw[uid] = float(np.mean(v)) if v else float("nan")
    if B50.exists():
        rows = [json.loads(l) for l in B50.read_text().splitlines()]
        for uid, _, _ in FIT_UNITS:
            v = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
                 and r["view"] == "srl" and r["model"] == "gbdt"
                 and r["normalize"] == "author_minmax" and r["cap"] == CONTEXT_CAP]
            gbdt_srl[uid] = float(np.median(v)) if v else float("nan")
    return gbdt_raw, gbdt_srl


# ---------------------------------------------------------------- main flow --
def main():
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    recs = []
    for uid, _, _ in FIT_UNITS:
        unit = load_fit_unit(*umap[uid])
        all_cells = unit.cells
        test_cells = unit.test_cells

        c_macro, c_pc, c_fin = coulomb_ref(test_cells)
        c_macro_all, _, _ = coulomb_ref(all_cells)
        N, degraded = pick_N(all_cells)
        s_macro, s_pc, s_term = severson_ref(unit.train_cells, test_cells, N)

        recs.append({"unit": uid, "ref": "coulomb", "n_test_cells": len(test_cells),
                     "rmse_cell_macro": c_macro, "rmse_cell_macro_allcells": c_macro_all,
                     "finite": c_fin, "per_cell": c_pc})
        recs.append({"unit": uid, "ref": "severson", "N": N, "N_degraded": degraded,
                     "n_test_cells": len(test_cells), "rmse_cell_macro": s_macro,
                     "terminal_abs_macro": s_term, "per_cell": s_pc})
        print(f"{uid:<16}coulomb macro={c_macro:.5f}  "
              f"severson(N={N:>3}) macro={s_macro:.5f}", flush=True)

    with OUT.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    report(recs)


def report(recs=None):
    recs = recs or [json.loads(l) for l in OUT.read_text().splitlines()]
    gbdt_raw, gbdt_srl = load_refs()
    by = {(r["unit"], r["ref"]): r for r in recs}

    print(f"\n{'='*94}\nminimal physical references vs GBDT (per-cell RMSE, lower is better; held-out-cell macro mean)")
    print(f"{'unit':<16}{'coulomb':>10}{'severson':>10}{'sev-N':>7}"
          f"{'GBDT-raw':>10}{'GBDT-SRL':>10}")
    c_vals, s_vals, gr_vals, gs_vals = [], [], [], []
    for uid, _, _ in FIT_UNITS:
        c = by[(uid, "coulomb")]["rmse_cell_macro"]
        s = by[(uid, "severson")]
        gr, gs = gbdt_raw.get(uid, np.nan), gbdt_srl.get(uid, np.nan)
        c_vals.append(c); s_vals.append(s["rmse_cell_macro"])
        gr_vals.append(gr); gs_vals.append(gs)
        print(f"{uid:<16}{c:>10.5f}{s['rmse_cell_macro']:>10.5f}{s['N']:>7}"
              f"{gr:>10.5f}{gs:>10.5f}")

    n_c = sum(1 for v in c_vals if np.isfinite(v))
    n_s = sum(1 for v in s_vals if np.isfinite(v))
    med_c = float(np.nanmedian(c_vals)); med_s = float(np.nanmedian(s_vals))
    med_gs = float(np.nanmedian(gs_vals))
    print(f"\n  finite units: coulomb {n_c}/11  severson {n_s}/11")
    print(f"  unit median error: coulomb {med_c:.5f}  severson {med_s:.5f}  "
          f"GBDT-SRL {med_gs:.5f}  GBDT-raw "
          f"{float(np.nanmedian(gr_vals)):.5f}")
    fin_ok = n_c >= 7 and n_s >= 7
    hi_ok = (med_c > med_gs) and (med_s > med_gs)
    print(f"  B1 finiteness (each >=7/11): {'holds' if fin_ok else 'falsified'}")
    print(f"  B1 higher median error (both refs > GBDT-SRL): {'holds' if hi_ok else 'falsified'}")
    print(f"  B1 overall verdict: {'holds' if (fin_ok and hi_ok) else 'falsified (see above; counter-examples reported as measured)'}")
    # counter-example details
    for uid, _, _ in FIT_UNITS:
        c = by[(uid, "coulomb")]["rmse_cell_macro"]
        s = by[(uid, "severson")]["rmse_cell_macro"]
        gs = gbdt_srl.get(uid, np.nan)
        bad = []
        if np.isfinite(c) and np.isfinite(gs) and c <= gs:
            bad.append("coulomb<=GBDT-SRL")
        if np.isfinite(s) and np.isfinite(gs) and s <= gs:
            bad.append("severson<=GBDT-SRL")
        if bad:
            print(f"    ! counter-example {uid}: {', '.join(bad)} "
                  f"(c={c:.5f} s={s:.5f} GBDT-SRL={gs:.5f})")


if __name__ == "__main__":
    main()
