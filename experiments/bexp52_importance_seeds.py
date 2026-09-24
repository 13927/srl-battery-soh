"""bexp52: multi-seed permutation importance by feature family for Table 10 -- response to R1-M12.

Background (reviewer R1-M12):
  Table 10 (permutation importance by feature family) currently comes only from the archive
  `results/battery/p6_importance.json`, which is a **single point estimate with no seed field**,
  and whose **generating script is not in the repository**. The reviewer singled out XJTU-R3:
  life-fraction cycle 34.2% vs the origin-self offset (from_first) 30.8% "nearly a tie", noting
  that with correlated inputs permutation importance can flip across seeds.

What this script does:
  - Protocol: the full 81-dim history stack (raw 16 + cycle 1 + lag1 16 + diff1 16 + from_first 16
    + roll5 16), author-protocol author_minmax normalization (cycle via cycle_author_minmax), GBDT
    (HistGradientBoostingRegressor, default parameters, identical to bexp14), context cap=2048.
  - Permutation importance is computed per unit on the **held-out cells**, with scoring on the
    -rmse_cell_macro scale (consistent with the existing implementation:
    sklearn.inspection.permutation_importance, n_repeats=5).
  - Family aggregation: **sum** the per-column importances within a family (columns first clipped
    at >=0), then normalize the six families to a 100% total.
  - Scale: 11 units x 10 seeds (0..9).

Consistency check (against the archived p6_importance.json):
  For every unit and family, compute |this experiment's multi-seed mean - archived value| (percentage
  points), report the maximum absolute deviation, and list the (unit, family) entries with deviation
  > 5 percentage points.

================== decision criteria (recorded in this docstring; prior registration is not claimed -- see Section 2.7) ==================
C1: if the maximum absolute deviation between this experiment's multi-seed means and the archived
    `p6_importance.json`, per unit and per family, is **<= 10 percentage points**:
      the archived value and the multi-seed mean are then judged to be of the **same magnitude**; the
      paper may report the multi-seed mean and dispersion for Table 10 as [multi-seed mean +/- std],
      noting that the archived single-run value lies within that magnitude.
    Otherwise:
      report only this experiment's multi-seed values in the paper, and **note the difference from
      the archived single run**.
Note: this script **does not tune parameters to match the archive**; the protocol (normalization /
      view / cap / GBDT) is fixed before the run, and deviations from the archive are reported as
      protocol/implementation differences.
==================================================================

Run:
./.venv/bin/python experiments/bexp52_importance_seeds.py            # full run, idempotent resume
./.venv/bin/python experiments/bexp52_importance_seeds.py --report   # reprint the report only
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, subsample_context
from battery_lab.temporal_features import build_view

OUT = ROOT / "results/battery/bexp52_importance.jsonl"
SUM = ROOT / "results/battery/bexp52_importance.json"
ARCHIVE = ROOT / "results/battery/p6_importance.json"
SEEDS = list(range(10))          # 10 seeds (>= 5 minimum; take the full set as time allows)
N_REPEATS = 5
CAP = CONTEXT_CAP                # 2048, consistent with the frozen convention of bexp14 / bexp42
FAMILIES = ["raw", "cycle", "lag1", "diff1", "from_first", "roll5"]
# 81-dim history-stack column split: [Z(16), cyc(1), lag1(16), diff1(16), from_first(16), roll5(16)]
FAM_SLICES = {
    "raw": slice(0, 16), "cycle": slice(16, 17), "lag1": slice(17, 33),
    "diff1": slice(33, 49), "from_first": slice(49, 65), "roll5": slice(65, 81),
}


def assemble(cells, Xnorm, cycle_mode):
    """Concatenate the history-view feature matrix; return X, y, and the per-row cell_id."""
    xs, ys, ids = [], [], []
    for c in cells:
        Xv = build_view(Xnorm[c.cell_id], "history", cycle_mode=cycle_mode)
        xs.append(Xv)
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def per_family_pct(imp_mean):
    """Per-column permutation importance -> family aggregate (column sums, clipped at >=0) -> normalized to 100%."""
    raw_sum = {f: float(np.clip(imp_mean[sl], 0.0, None).sum())
               for f, sl in FAM_SLICES.items()}
    total = sum(raw_sum.values())
    if total <= 0:
        return {f: 0.0 for f in FAMILIES}, raw_sum
    return {f: raw_sum[f] / total * 100.0 for f in FAMILIES}, raw_sum


def run_unit_seed(unit, seed):
    """One (unit, seed): fit GBDT -> held-out-cell permutation importance -> per-family percentages."""
    Xn = normalize_cells(unit.cells, "author_minmax")
    cycle_mode = "author_minmax"
    X_tr, y_tr, _ = assemble(unit.train_cells, Xn, cycle_mode)
    X_te, y_te, te_ids = assemble(unit.test_cells, Xn, cycle_mode)
    Xc, yc = subsample_context(X_tr, y_tr, CAP, seed)

    t0 = time.time()
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(Xc, yc)

    # same rmse_cell_macro convention as the existing code: macro mean of per-held-out-cell RMSE
    def scorer(estimator, X, y):
        pred = estimator.predict(X)
        errs = [np.sqrt(np.mean((y[te_ids == cid] - pred[te_ids == cid]) ** 2))
                for cid in np.unique(te_ids)]
        return -float(np.mean(errs))

    res = permutation_importance(model, X_te, y_te, scoring=scorer,
                                 n_repeats=N_REPEATS, random_state=seed,
                                 n_jobs=1)
    fam_pct, fam_raw = per_family_pct(res.importances_mean)
    base = scorer(model, X_te, y_te)          # = -rmse_cell_macro
    return {
        "unit": unit.unit_id, "seed": seed, "cap": CAP, "n_repeats": N_REPEATS,
        "n_test_cells": len(unit.test_cells), "n_test_rows": int(len(y_te)),
        "rmse_cell_macro": float(-base),
        "families": {f: round(fam_pct[f], 4) for f in FAMILIES},
        "family_raw_sum": {f: fam_raw[f] for f in FAMILIES},
        "runtime_s": time.time() - t0,
    }


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["seed"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        ds, bk = umap[uid]
        unit = load_fit_unit(ds, bk)
        for seed in args.seeds:
            if (uid, seed) in done:
                continue
            r = run_unit_seed(unit, seed)
            with OUT.open("a") as f:
                f.write(json.dumps(r) + "\n")
            fam = r["families"]
            print(f"{uid:<15}seed={seed} macro={r['rmse_cell_macro']:.5f} "
                  f"raw={fam['raw']:5.1f} cyc={fam['cycle']:5.1f} "
                  f"lag1={fam['lag1']:5.1f} ff={fam['from_first']:5.1f} "
                  f"[{r['runtime_s']:.1f}s]", flush=True)
    summarize()
    report()


def summarize():
    """jsonl -> json summary: per unit and family mean/std/min/max + tie flags."""
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    summary = {}
    for uid, _, _ in FIT_UNITS:
        us = [r for r in rows if r["unit"] == uid]
        if not us:
            continue
        fams = {"n_seeds": len(us)}
        for f in FAMILIES:
            v = np.array([r["families"][f] for r in us], dtype=float)
            fams[f] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                       "min": float(v.min()), "max": float(v.max())}
        # tie analysis: sort by mean descending, inspect adjacent gaps, find the "closest" and
        # "second-closest" family pairs
        order = sorted(FAMILIES, key=lambda f: -fams[f]["mean"])
        gaps = []
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            gaps.append({"pair": [a, b],
                         "gap": fams[a]["mean"] - fams[b]["mean"],
                         "combined_std": fams[a]["std"] + fams[b]["std"],
                         "range_overlap": not (fams[a]["min"] > fams[b]["max"] or
                                               fams[b]["min"] > fams[a]["max"])})
        gaps_sorted = sorted(gaps, key=lambda g: g["gap"])
        closest = gaps_sorted[0] if gaps_sorted else None
        second = gaps_sorted[1] if len(gaps_sorted) > 1 else None
        for g in (closest, second):
            if g is not None:
                g["tie_within_error"] = bool(g["gap"] <= g["combined_std"]) or g["range_overlap"]
        fams["top2"] = [order[0], order[1]]
        fams["closest_pair"] = closest
        fams["second_closest_pair"] = second
        summary[uid] = fams

    # comparison against the archive
    arch = json.load(ARCHIVE.open()) if ARCHIVE.exists() else {}
    dev = []
    for uid in summary:
        if uid not in arch:
            continue
        for f in FAMILIES:
            d = abs(summary[uid][f]["mean"] - arch[uid][f])
            dev.append((uid, f, summary[uid][f]["mean"], arch[uid][f], d))
    max_dev = max((x[4] for x in dev), default=float("nan"))
    big = [x for x in dev if x[4] > 5.0]
    out = {"_meta": {"seeds": len({r["seed"] for r in rows}), "cap": CAP,
                     "n_repeats": N_REPEATS, "families": FAMILIES,
                     "max_abs_dev_vs_archive_pp": max_dev,
                     "n_cells_dev_gt_5pp": len(big),
                     "criterion_C1_ok": bool(np.isfinite(max_dev) and max_dev <= 10.0)},
           "per_unit": summary,
           "dev_vs_archive": [{"unit": u, "family": f, "ours": o, "archive": a,
                               "abs_dev_pp": d} for u, f, o, a, d in dev]}
    json.dump(out, SUM.open("w"), indent=1)
    return out


def report():
    out = json.load(SUM.open())
    per = out["per_unit"]
    print(f"\n{'='*100}")
    print(f"bexp52 multi-seed permutation importance (per family %, mean+/-std over {out['_meta']['seeds']} seeds, cap={CAP})")
    hdr = f"{'unit':<15}" + "".join(f"{f:>13}" for f in FAMILIES) + f"{'  tie (2nd closest pair)':>0}"
    print(hdr)
    for uid, _, _ in FIT_UNITS:
        if uid not in per:
            continue
        row = f"{uid:<15}"
        for f in FAMILIES:
            row += f"{per[uid][f]['mean']:>6.1f}±{per[uid][f]['std']:<5.1f}"
        sc = per[uid]["second_closest_pair"]
        pair = "/".join(sc["pair"]) if sc else "-"
        mark = "tie" if (sc and sc["tie_within_error"]) else "no"
        row += f"  {pair} gap={sc['gap']:.1f} {mark}" if sc else "  -"
        print(row)

    print(f"\n--- consistency against archived p6_importance.json ---")
    print(f"maximum absolute deviation = {out['_meta']['max_abs_dev_vs_archive_pp']:.2f} percentage points")
    big = [d for d in out["dev_vs_archive"] if d["abs_dev_pp"] > 5.0]
    if big:
        print(f"(unit, family) with deviation > 5 percentage points -- {len(big)} entries:")
        for d in sorted(big, key=lambda x: -x["abs_dev_pp"]):
            print(f"  {d['unit']:<15}{d['family']:<12}ours={d['ours']:6.1f} "
                  f"archive={d['archive']:6.1f} |dev|={d['abs_dev_pp']:5.1f}")
    else:
        print("no (unit, family) with deviation > 5 percentage points.")

    ok = out["_meta"]["criterion_C1_ok"]
    print(f"\n=== frozen verdict C1 ===")
    print(f"  max absolute deviation {out['_meta']['max_abs_dev_vs_archive_pp']:.2f}pp "
          f"{'<=' if ok else '>'} 10pp -> "
          f"{'holds: archive and multi-seed means are the same magnitude; usable for Table 10' if ok else 'fails: report only this experiment multi-seed values plus the difference'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
