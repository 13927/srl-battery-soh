"""bexp50: linear baselines -- the first item of the "bottom-of-the-ladder simple
baselines" requested by Reviewer 1 (R1-M5).

Motivation:
  1. The paper's lowest baseline so far was GBDT/TabPFN/MLP; linear regression had
     never been run. The reviewer explicitly asked for "ridge (or linear)
     regression on compact SRL in both pipelines".
  2. Key purpose: separate the representation contribution (raw -> compact SRL)
     from the model-family contribution (Ridge <-> GBDT). Swap the model family
     under one representation and the representation under one model family, then
     read the four cells side by side.

Models: sklearn.linear_model.Ridge(alpha=1.0, library default) as the main arm;
      LinearRegression as the secondary arm.
      * Do NOT call TabPFN (no weights on this machine; it will fail).
        GBDT is reused only as a same-level cross arm.
Features: raw (17 dims = 16 statistics + cycle) and compact SRL
      (33 dims = 16 + cycle + 16 from_first). Both reuse the existing battery_lab
      implementations (build_view / add_history_features).
Pipelines: both are run --
      Pipeline A = normalize_cells("author_minmax"): per-cell full-lifetime min-max + cycle_author_minmax;
      Pipeline B = normalize_cells("source"):       statistics fitted on the source cell pool + cycle_scaled(/200).
      Field-for-field identical to protocols.evaluate_unit.
Scale: 11 units x 5 seeds (0-4) x {raw, srl} x {ridge, linear, gbdt} x {A, B}.

================== decision criteria (recorded in this docstring; prior registration is not claimed -- see Section 2.7) ==================
A1 (representation contribution): the median per-unit Delta% of compact SRL relative to
     raw in the same pipeline must be < 0 in at least one of the two pipelines under
     Ridge (i.e. the representation contribution is not adverse even in the most
     strongly regularised linear family). Whether it holds or not, report the
     per-configuration Delta% of all 11 units as measured.
A2 (environment reproducibility, hard check): re-run GBDT with this script's same
    harness on XJTU-2C / raw / seed 0 / author_minmax / cap=2048 and compare with
    bexp14's archived rmse_cell_macro = 0.004917527799087213. |deviation| > 1% must be
    flagged prominently in the report; it affects the comparability of all new numbers
    (Ridge produced under sklearn 1.9.1) with bexp14.
========================================================================

Key notes:
  - bexp14_gbdt.jsonl has only the raw / history views and no compact-SRL arm.
    "GBDT-SRL" therefore cannot be read from the archive; this script re-runs the GBDT
    compact-SRL arm under the **same harness, same cap, same seeds** so that the
    representation/model-family decomposition is a strictly common-protocol contrast
    (this is the purpose of the A2 reproducibility check: first show that this harness
    can reproduce bexp14 bit for bit, then trust its srl numbers).
  - Idempotent: completed (unit, view, model, seed, normalize, cap) cells are read from
    the result file and skipped.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import (CONTEXT_CAP, fit_predict_gbdt,
                                   subsample_context)
from battery_lab.temporal_features import add_history_features, build_view

OUT = ROOT / "results/battery/bexp50_linear.jsonl"
SEEDS = [0, 1, 2, 3, 4]
VIEWS = ["raw", "srl"]
NORMZ = ["author_minmax", "source"]
GBDT_ARCHIVED = 0.004917527799087213   # bexp14: XJTU-2C|raw|seed0|author_minmax|cap2048
GBDT_ARCHIVED_KEY = ("XJTU-2C", "raw", "gbdt", 0, "author_minmax", CONTEXT_CAP)


# ------------------------------------------------------------------ models ---
def fit_predict_ridge(X_tr, y_tr, X_te, seed):
    """Ridge, library default alpha=1.0 (untuned, symmetric with GBDT/TabPFN being untuned)."""
    m = Ridge(alpha=1.0)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def fit_predict_linear(X_tr, y_tr, X_te, seed):
    """Ordinary least-squares linear regression (no regularisation)."""
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


MODELS = {"ridge": fit_predict_ridge, "linear": fit_predict_linear,
          "gbdt": fit_predict_gbdt}


# ------------------------------------------------------------------ features --
def build_xy(Z, view, cycle_mode):
    """Assemble the view matrix for one cell.

    raw: [Z(16), cycle(1)] = 17 dims (= the existing build_view implementation)
    srl: [Z(16), cycle(1), from_first(16)] = 33 dims (from_first comes from the
         existing add_history_features, i.e. Z - Z[0], consistent with the compact
         SRL of bexp33/34/35)
    """
    base = build_view(Z, "raw", cycle_mode=cycle_mode)
    if view == "raw":
        return base
    if view == "history":
        # Full 81-dim history stack (= the existing build_view implementation);
        # used by the bexp56 strict-online re-run. No effect on the raw/srl branches.
        return build_view(Z, "history", cycle_mode=cycle_mode)
    if view == "srl":
        d = Z.shape[1]
        from_first = add_history_features(Z)[:, 2 * d:3 * d]
        return np.hstack([base, from_first])
    raise ValueError(f"unknown view {view!r}")


def assemble(cells, Xn, view, cycle_mode):
    xs, ys, ids = [], [], []
    for c in cells:
        xs.append(build_xy(Xn[c.cell_id], view, cycle_mode))
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def evaluate(unit, view, model, seed, normalize, cap=CONTEXT_CAP):
    """Reproduce the matrix assembly of protocols.evaluate_unit, but support view='srl' and linear models."""
    tr, te = unit.train_cells, unit.test_cells
    if normalize == "author_minmax":
        Xn = normalize_cells(tr + te, "author_minmax")
        cycle_mode = "author_minmax"
    else:
        Xn = normalize_cells(tr + te, "source", source_cells=tr)
        cycle_mode = "scaled200"

    X_tr, y_tr, _ = assemble(tr, Xn, view, cycle_mode)
    X_te, y_te, te_ids = assemble(te, Xn, view, cycle_mode)
    X_tr, y_tr = subsample_context(X_tr, y_tr, cap, seed)

    t0 = time.time()
    pred = MODELS[model](X_tr, y_tr, X_te, seed)
    runtime = time.time() - t0

    overall = float(np.sqrt(np.mean((y_te - pred) ** 2)))
    per_cell = {cid: float(np.sqrt(np.mean((y_te[te_ids == cid]
                                            - pred[te_ids == cid]) ** 2)))
                for cid in np.unique(te_ids)}
    return {
        "unit": unit.unit_id, "view": view, "model": model, "seed": seed,
        "normalize": normalize, "cap": cap,
        "n_train_rows": int(len(y_tr)), "n_test_rows": int(len(y_te)),
        "n_train_cells": len(tr), "n_test_cells": len(te),
        "rmse_overall": overall,
        "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
        "per_cell": per_cell,
        "runtime_s": runtime,
    }


# ----------------------------------------------------------------- main flow -
def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["view"], r["model"], r["seed"],
                      r["normalize"], r["cap"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--models", nargs="+", default=["ridge", "linear", "gbdt"])
    ap.add_argument("--views", nargs="+", default=VIEWS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        unit = load_fit_unit(*umap[uid])
        for normalize in NORMZ:
            for view in args.views:
                for model in args.models:
                    for seed in args.seeds:
                        key = (uid, view, model, seed, normalize, CONTEXT_CAP)
                        if key in done:
                            continue
                        t0 = time.time()
                        r = evaluate(unit, view, model, seed, normalize)
                        with OUT.open("a") as f:
                            f.write(json.dumps(r) + "\n")
                        print(f"{uid:<15}{normalize[:6]:<7}{view:<4}{model:<7}"
                              f"seed={seed} macro={r['rmse_cell_macro']:.6f} "
                              f"overall={r['rmse_overall']:.6f} "
                              f"[{time.time()-t0:.2f}s]", flush=True)
    report()


# ------------------------------------------------------------------ report ---
def _macro(rows, uid, view, model, normalize):
    v = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
         and r["view"] == view and r["model"] == model
         and r["normalize"] == normalize]
    return float(np.median(v)) if v else np.nan


def repro_check(rows):
    """A2: this harness's GBDT archived cell vs bexp14."""
    hit = [r for r in rows
           if (r["unit"], r["view"], r["model"], r["seed"], r["normalize"],
               r["cap"]) == GBDT_ARCHIVED_KEY]
    print(f"\n{'='*78}\nA2 environment reproducibility check (GBDT / XJTU-2C / raw / seed0 / "
          f"author_minmax / cap2048)")
    if not hit:
        print("  cell not found; cannot reproduce (check that the gbdt arm was run)")
        return None
    got = hit[0]["rmse_cell_macro"]
    dev = (got - GBDT_ARCHIVED) / GBDT_ARCHIVED * 100
    flag = "[>1% must be flagged prominently]" if abs(dev) > 1 else "[OK]"
    print(f"  archived bexp14: {GBDT_ARCHIVED:.9f}")
    print(f"  this re-run    : {got:.9f}")
    print(f"  deviation      : {dev:+.4f}%  {flag}")
    return {"archived": GBDT_ARCHIVED, "rerun": got, "dev_pct": dev,
            "ok": bool(abs(dev) <= 1)}


def report():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    if not rows:
        return
    repro = repro_check(rows)

    for normalize in NORMZ:
        tag = "A: author_minmax (per-cell full-lifetime min-max)" if normalize == "author_minmax" \
            else "B: source (source-fitted + scaled cycle)"
        print(f"\n{'='*92}\npipeline {tag} -- rmse_cell_macro (median over 5 seeds)")
        print(f"{'unit':<16}{'R-raw':>10}{'R-srl':>10}{'RDelta%':>8}"
              f"{'G-raw':>10}{'G-srl':>10}{'GDelta%':>8}{'L-srl':>10}")
        d_r, d_g = [], []
        for uid, _, _ in FIT_UNITS:
            rr = _macro(rows, uid, "raw", "ridge", normalize)
            rs = _macro(rows, uid, "srl", "ridge", normalize)
            gr = _macro(rows, uid, "raw", "gbdt", normalize)
            gs = _macro(rows, uid, "srl", "gbdt", normalize)
            ls = _macro(rows, uid, "srl", "linear", normalize)
            dr = (rs - rr) / rr * 100 if np.isfinite(rr) and np.isfinite(rs) else np.nan
            dg = (gs - gr) / gr * 100 if np.isfinite(gr) and np.isfinite(gs) else np.nan
            d_r.append(dr); d_g.append(dg)
            print(f"{uid:<16}{rr:>10.6f}{rs:>10.6f}{dr:>+8.1f}"
                  f"{gr:>10.6f}{gs:>10.6f}{dg:>+8.1f}{ls:>10.6f}")
        med_r = float(np.nanmedian(d_r)); med_g = float(np.nanmedian(d_g))
        print(f"  median Delta%:  Ridge {med_r:+.2f}%   GBDT {med_g:+.2f}%   "
              f"| units improved: Ridge {sum(1 for d in d_r if d<0)}/11  "
              f"GBDT {sum(1 for d in d_g if d<0)}/11")

    # A1 verdict
    med = {}
    for normalize in NORMZ:
        ds = []
        for uid, _, _ in FIT_UNITS:
            rr = _macro(rows, uid, "raw", "ridge", normalize)
            rs = _macro(rows, uid, "srl", "ridge", normalize)
            if np.isfinite(rr) and np.isfinite(rs):
                ds.append((rs - rr) / rr * 100)
        med[normalize] = float(np.nanmedian(ds))
    ok = min(med.values()) < 0
    print(f"\n{'='*78}\nA1 verdict: Ridge SRL median Delta% "
          f"[A={med['author_minmax']:+.2f}%, B={med['source']:+.2f}%] "
          f"-> {'holds (at least one <0)' if ok else 'falsified (both >=0)'}")
    if repro and not repro["ok"]:
        print("  !! note: A2 reproduction deviation >1%; comparability of the numbers above with bexp14 is limited -- see the red flag in the report.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
