"""bexp22: the ensemble branch of source-side selection (ens-mean, plus a
Caruana-style selection over a library that includes the ensemble).

================ frozen criteria (committed before the run) ================
E1: ens-mean (the average of the TabPFN-history and GBDT-history predictions) beats
    both members in >= 6 of 11 units -> the ensemble enters the configuration
    library; otherwise selection alone is used
E2 (bias-diversity identity check): ens RMSE <= the mean of the two member RMSEs,
    which should hold 11/11 (a mathematical property; a violation is an
    implementation bug)
=======================================================

A. Test side: per unit x 3 seeds, the TabPFN-history and GBDT-history models each
   predict the official test cells; their average is ens-mean, compared with the
   members (for scoring only, taking no part in selection).
B. Source side: on the same subsampled source-cell folds as bexp21 (same
   SAMPLING_SEED), compute the source-side leave-one-cell-out estimate of ens ->
   select over the five-config library {4 single configs + ens} with
   select_config (the smallest instance of Caruana library selection: the library
   holds the ensemble candidate, the validation set is source-side LOCO).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import (CONTEXT_CAP, assemble_matrix,
                                   fit_predict_gbdt, fit_predict_tabpfn,
                                   subsample_context)
from battery_lab.source_selection import MAX_TABPFN_CELLS, subsample_cells

OUT_TEST = Path("results/battery/bexp22_ensemble_test.json")
OUT_SRC = Path("results/battery/bexp22_ensemble_source.json")
SEEDS = [0, 1, 2]


def _prep(cells_tr, cells_te, view):
    Xn = normalize_cells(cells_tr + cells_te, "author_minmax")
    X_tr, y_tr, _ = assemble_matrix(cells_tr, Xn, view, "author_minmax")
    X_te, y_te, te_ids = assemble_matrix(cells_te, Xn, view, "author_minmax")
    return X_tr, y_tr, X_te, y_te, te_ids


def duo_predict(cells_tr, cells_te, seed):
    """Predictions of TabPFN-history and GBDT-history (same folds, same budget),
    returning (pred_t, pred_g, y)."""
    X_tr, y_tr, X_te, y_te, _ = _prep(cells_tr, cells_te, "history")
    Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
    pt = fit_predict_tabpfn(Xc, yc, X_te, seed)
    pg = fit_predict_gbdt(Xc, yc, X_te, seed)
    return pt, pg, y_te


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_test_side(units, state):
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in units:
        if uid in state:
            continue
        unit = load_fit_unit(*umap[uid])
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            pt, pg, y = duo_predict(unit.train_cells, unit.test_cells, seed)
            pe = (np.asarray(pt) + np.asarray(pg)) / 2
            per_seed.append({"seed": seed, "tab": rmse(pt, y),
                             "gbdt": rmse(pg, y), "ens": rmse(pe, y),
                             "runtime_s": time.time() - t0})
            print(f"[test] {uid:<15}seed={seed} tab={per_seed[-1]['tab']:.6f} "
                  f"gbdt={per_seed[-1]['gbdt']:.6f} ens={per_seed[-1]['ens']:.6f} "
                  f"[{per_seed[-1]['runtime_s']:.0f}s]", flush=True)
        state[uid] = {
            "tab": float(np.mean([s["tab"] for s in per_seed])),
            "gbdt": float(np.mean([s["gbdt"] for s in per_seed])),
            "ens": float(np.mean([s["ens"] for s in per_seed])),
            "per_seed": per_seed,
        }
        json.dump(state, OUT_TEST.open("w"), indent=1)
    return state


def run_source_side(units, state):
    """Source-side leave-one-cell-out estimate of ens (same subsampled folds as
    bexp21)."""
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in units:
        if uid in state:
            continue
        unit = load_fit_unit(*umap[uid])
        cells = subsample_cells(list(unit.train_cells), MAX_TABPFN_CELLS)
        per = []
        t0 = time.time()
        for i, tgt in enumerate(cells):
            srcs = [c for j, c in enumerate(cells) if j != i]
            for seed in SEEDS:
                pt, pg, y = duo_predict(srcs, [tgt], seed)
                pe = (np.asarray(pt) + np.asarray(pg)) / 2
                per.append({"cell": tgt.cell_id, "seed": seed,
                            "ens": rmse(pe, y)})
        cm = {}
        for p in per:
            cm.setdefault(p["cell"], []).append(p["ens"])
        arr = np.array([np.mean(v) for v in cm.values()])
        state[uid] = {"est_macro": float(arr.mean()),
                      "se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
                      "n_cells": len(cells), "per": per,
                      "runtime_s": time.time() - t0}
        json.dump(state, OUT_SRC.open("w"), indent=1)
        print(f"[src ] {uid:<15} ens est={state[uid]['est_macro']:.6f} "
              f"se={state[uid]['se']:.6f} [{state[uid]['runtime_s']:.0f}s]",
              flush=True)
    return state


def report():
    if not OUT_TEST.exists():
        print("(no test-side data)")
        return
    t = json.load(OUT_TEST.open())
    print(f"\n{'='*70}\nE1/E2 verdict (test side, mean of 3 seeds, overall RMSE)")
    print(f"{'unit':<16}{'tab-hist':>10}{'gbdt-hist':>10}{'ens':>10}{'beats both?':>12}")
    win = kv_ok = 0
    for uid, d in t.items():
        w = d["ens"] < min(d["tab"], d["gbdt"])
        kv = d["ens"] <= (d["tab"] + d["gbdt"]) / 2 + 1e-12
        win += int(w)
        kv_ok += int(kv)
        print(f"{uid:<16}{d['tab']:>10.6f}{d['gbdt']:>10.6f}{d['ens']:>10.6f}"
              f"{'✓' if w else '':>9}")
    n = len(t)
    print(f"\n  E1 (ens beats both members in >=6/11): {win}/{n} -> "
          + ("holds -- the ensemble enters the library" if win >= 6
             else "falsified -- selection alone"))
    print(f"  E2 (bias-diversity identity 11/11): {kv_ok}/{n} -> "
          + ("holds" if kv_ok == n else "BUG"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["test", "source", "report"],
                    default="test")
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    a = ap.parse_args()
    if a.part == "report":
        report()
    elif a.part == "test":
        st = json.load(OUT_TEST.open()) if OUT_TEST.exists() else {}
        run_test_side(a.units, st)
        report()
    else:
        st = json.load(OUT_SRC.open()) if OUT_SRC.exists() else {}
        run_source_side(a.units, st)
