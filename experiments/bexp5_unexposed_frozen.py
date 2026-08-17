"""bexp5: frozen verification on the unexposed library.

The unexposed library is MATR-CLO (the closed-loop-optimisation batch of Attia et
al. 2020, 2019-01-24):
  - not among the three MIT batches used by PINN4SOH
    (2017-05-12 / 06-30 / 2018-04-12)
  - not in the MATR Batch 1 used during development
  - 45 cells / 39526 cycle rows / 28 strict base features (MATR style)

Strict leave-one-cell-out protocol:
  - normalisation statistics are fitted on source cells only; target cells are
    merely transformed
  - the cycle-position feature is cycle/200 (not the reference full-lifetime
    min-max)
  - imputation is history-only (source-group medians, forward fill within the
    target stream, plus missingness indicators)

Frozen method set: {raw TabPFN, history TabPFN, rule output} x 5 seeds
Metric: macro RMSE over cells (primary) plus a cell-level bootstrap 95 per cent CI

This script is run once on this library; the result is reported as measured,
whether it supports the rule or not.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.protocols import CONTEXT_CAP, subsample_context
from battery_lab.temporal_features import build_view
from battery_lab.data_adapters import CellData, FitUnit
from battery_lab.unexposed_adapters import (ImputationRule, apply_rule,
                                            precompute_ffill)
from battery_lab.view_selection import (MAX_SOURCE_CELLS, SAMPLING_SEED,
                                        decide_from_per_cell)

NPZ = Path("external/unexposed/matr_clo_unit.npz")
OUT = Path("results/battery/bexp5_unexposed_frozen.json")
SEEDS = [0, 1, 2, 3, 4]


_FFILL_CACHE = {}


def load_unit_npz(path=None):
    """Load a FitUnit from npz (instead of pickle, so nothing is executed on
    deserialisation)."""
    path = path or NPZ
    z = np.load(path, allow_pickle=False)
    cells = []
    for cid in [str(x) for x in z["cell_ids"]]:
        cells.append(CellData(cell_id=cid, X=z[f"X_{cid}"], y=z[f"y_{cid}"],
                              is_test=bool(z[f"t_{cid}"])))
    return FitUnit(unit_id=str(z["unit_id"]), dataset=str(z["dataset"]),
                   cells=cells)


def get_ffill(cell):
    """Within-cell forward fill (independent of the source group, so it is cached
    once globally)."""
    if cell.cell_id not in _FFILL_CACHE:
        _FFILL_CACHE[cell.cell_id] = precompute_ffill(cell.X)
    return _FFILL_CACHE[cell.cell_id]


def prepare(unit, source_cells, target_cells):
    """Strict-mode feature preparation: the imputation and normalisation rules are
    fitted on the source group, target cells are only transformed."""
    rule = ImputationRule.fit(np.vstack([c.X for c in source_cells]))
    imputed, indicators = {}, {}
    for c in source_cells + target_cells:
        X_ff, still_nan, miss = get_ffill(c)
        imputed[c.cell_id] = apply_rule(X_ff, still_nan, rule)
        indicators[c.cell_id] = miss.astype(float)
    pool = np.vstack([imputed[c.cell_id] for c in source_cells])
    mn, mx = pool.min(0), pool.max(0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    normed = {k: 2 * (v - mn) / rng - 1 for k, v in imputed.items()}
    return normed, indicators


def assemble(cells, normed, indicators, view):
    xs, ys, ids = [], [], []
    for c in cells:
        Xv = build_view(normed[c.cell_id], view, cycle_mode="scaled200")
        # missingness indicator (collapsed to one column by the row mean, to avoid
        # doubling the width)
        ind = indicators[c.cell_id].mean(axis=1, keepdims=True)
        xs.append(np.hstack([Xv, ind]))
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def run_view(unit, view, seed, source_cells=None, target_cells=None):
    from residual_lab.models import make_model
    src = source_cells if source_cells is not None else unit.train_cells
    tgt = target_cells if target_cells is not None else unit.test_cells
    normed, ind = prepare(unit, src, tgt)
    X_tr, y_tr, _ = assemble(src, normed, ind, view)
    X_te, y_te, te_ids = assemble(tgt, normed, ind, view)
    X_tr, y_tr = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
    m = make_model("tabpfn_reg", seed=seed)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    per_cell = {cid: float(np.sqrt(np.mean((y_te[te_ids == cid]
                                           - pred[te_ids == cid]) ** 2)))
                for cid in np.unique(te_ids)}
    return per_cell, pred, y_te, te_ids


def rule_d_on_unexposed(unit):
    """The selection rule: nested leave-one-cell-out on source cells only (strict
    mode)."""
    cells = unit.train_cells
    if len(cells) > MAX_SOURCE_CELLS:
        rng = np.random.default_rng(SAMPLING_SEED)
        idx = sorted(rng.choice(len(cells), MAX_SOURCE_CELLS, replace=False))
        cells = [cells[i] for i in idx]
    per_seed = []
    for seed in SEEDS:
        pc = {"raw": {}, "history": {}}
        for i, tgt in enumerate(cells):
            srcs = [c for j, c in enumerate(cells) if j != i]
            for view in ("raw", "history"):
                res, _, _, _ = run_view(unit, view, seed, srcs, [tgt])
                pc[view][tgt.cell_id] = float(np.mean(list(res.values())))
        d = decide_from_per_cell(pc, boot_seed=seed)
        d["seed"] = seed
        per_seed.append(d)
        print(f"  ruleD seed={seed}: {d['decision']} diff={d['mean_diff']:+.6f} "
              f"CI={[round(x,6) for x in d['ci']]}", flush=True)
    votes = [d["decision"] for d in per_seed]
    counts = {v: votes.count(v) for v in ("raw", "history", "abstain")}
    top = max(counts.values())
    cand = [v for v, c in counts.items() if c == top]
    return (cand[0] if len(cand) == 1 else "abstain"), votes, per_seed


def bootstrap_ci(diff, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = [np.mean(diff[rng.integers(0, len(diff), len(diff))])
         for _ in range(n_boot)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    unit = load_unit_npz()
    print(f"unexposed library {unit.unit_id}: {len(unit.cells)} cells "
          f"(source {len(unit.train_cells)} / target {len(unit.test_cells)})",
          flush=True)

    result = {"unit": unit.unit_id, "n_cells": len(unit.cells),
              "n_source": len(unit.train_cells), "n_target": len(unit.test_cells)}

    # 1) the rule decides, from source data only
    t0 = time.time()
    print("\n[1/2] nested source-side selection (source cells only)", flush=True)
    decision, votes, per_seed = rule_d_on_unexposed(unit)
    result["ruleD"] = {"decision": decision, "votes": votes,
                       "per_seed": [{k: v for k, v in d.items()
                                     if k != "per_cell"} for d in per_seed],
                       "runtime_s": time.time() - t0}
    print(f"  -> decision: {decision}  votes={votes}", flush=True)

    # 2) target-side evaluation of the two fixed views
    print("\n[2/2] evaluation on target cells (strict leave-one-cell-out)",
          flush=True)
    per_view = {}
    for view in ("raw", "history"):
        macros, cell_tables = [], []
        for seed in SEEDS:
            pc, _, _, _ = run_view(unit, view, seed)
            macros.append(float(np.mean(list(pc.values()))))
            cell_tables.append(pc)
            print(f"  {view:<8} seed={seed} macro={macros[-1]:.6f}", flush=True)
        per_view[view] = {"macro_by_seed": macros,
                          "macro_mean": float(np.mean(macros)),
                          "per_cell_by_seed": cell_tables}
    result["views"] = per_view

    # 3) cell-level bootstrap of the difference between views
    cells = sorted(per_view["raw"]["per_cell_by_seed"][0])
    diff = np.array([
        np.mean([per_view["history"]["per_cell_by_seed"][s][c] for s in range(len(SEEDS))])
        - np.mean([per_view["raw"]["per_cell_by_seed"][s][c] for s in range(len(SEEDS))])
        for c in cells])
    result["history_minus_raw"] = {
        "mean": float(diff.mean()), "ci": bootstrap_ci(diff),
        "n_target_cells": len(cells)}

    # 4) how the rule output actually performs
    best_view = ("history" if per_view["history"]["macro_mean"]
                 < per_view["raw"]["macro_mean"] else "raw")
    if decision == "abstain":
        d_macro = (per_view["raw"]["macro_mean"]
                   + per_view["history"]["macro_mean"]) / 2  # conservative bound
        d_note = ("abstain -> the two views are averaged (reported as the "
                  "arithmetic-mean upper bound on RMSE)")
    else:
        d_macro = per_view[decision]["macro_mean"]
        d_note = f"selected {decision}"
    result["verdict"] = {
        "post_hoc_best_view": best_view,
        "ruleD_correct_direction": decision in ("abstain", best_view),
        "ruleD_hit": decision == best_view,
        "ruleD_macro": d_macro, "note": d_note,
        "raw_macro": per_view["raw"]["macro_mean"],
        "history_macro": per_view["history"]["macro_mean"],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, OUT.open("w"), indent=1)
    report(result)


def report(result=None):
    r = result or json.load(OUT.open())
    v = r["verdict"]
    print(f"\n===== frozen verification on the unexposed library: {r['unit']} =====")
    print(f"cells: {r['n_cells']} (source {r['n_source']} / "
          f"target {r['n_target']})")
    print(f"raw     macro RMSE = {v['raw_macro']:.6f}")
    print(f"history macro RMSE = {v['history_macro']:.6f}")
    hr = r["history_minus_raw"]
    print(f"history - raw = {hr['mean']:+.6f}  95% CI {[round(x,6) for x in hr['ci']]}"
          f"  (n={hr['n_target_cells']} target cells)")
    print(f"post-hoc best view = {v['post_hoc_best_view']}")
    print(f"rule decision = {r['ruleD']['decision']}  "
          f"votes = {r['ruleD']['votes']}")
    print(f"  direction correct (abstention counted) = "
          f"{v['ruleD_correct_direction']}   strict hit = {v['ruleD_hit']}")
    print(f"  macro RMSE of the rule output = {v['ruleD_macro']:.6f}  "
          f"({v['note']})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
