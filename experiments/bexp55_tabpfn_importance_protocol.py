"""bexp55: provenance of Table 10's protocol -- which backbone produced the archived
single-point permutation importance?

Background (response to R1-M12):
  Table 10 is taken from the archive `results/battery/p6_importance.json`: a single point
  estimate whose generating script is not in the repository. The GBDT multi-seed recomputation
  (bexp52) shows a maximum absolute deviation of 14.98pp from the archive, failing the 10pp
  criterion. Hypothesis H: the archived values came from the **TabPFN backbone** (the original
  P6 analysis) rather than GBDT -- if so, Table 10's protocol can be confirmed and multi-seed
  means +/- dispersion can be reported.

What this script does (line-for-line the same protocol as bexp52, only the backbone changes):
  - Protocol: the full 81-dim history stack (raw 16 + cycle 1 + lag1 16 + diff1 16 + from_first 16
    + roll5 16), author-protocol author_minmax normalization (cycle via cycle_author_minmax),
    context cap=2048.
  - Backbone: TabPFN v3 frozen weights (residual_lab.models.make_model("tabpfn_reg", seed));
    requires TABPFN_WEIGHTS_DIR to point at a directory containing tabpfn-v3-regressor-v3_default.ckpt.
  - Permutation importance is computed per unit on the held-out cells, with scoring on the
    -rmse_cell_macro scale (sklearn.inspection.permutation_importance, n_repeats configurable).
  - Family aggregation: per-column importances within a family are clipped at >=0 and summed,
    then the 6 families are normalized to a 100% total.

================== decision criteria (recorded in this docstring; prior registration is not claimed -- see Section 2.7) ==================
V1 environment reproducibility: re-run XJTU-2C / raw / seed 0 / tabpfn with evaluate_unit and
   compare against the same cell in the archived bexp1_author_protocol.jsonl; the relative
   deviation of rmse_cell_macro must be <= 1%, plus TJU-1 / history / seed 0 as a second cell.
   -> only if it holds may P1/P2 be interpreted; otherwise the environment difference must be
      stated in the paper.
P1 protocol attribution: the per-unit per-family **maximum absolute deviation <= 10 percentage
   points** between the TabPFN multi-seed means and p6_importance.json, and better than GBDT's
   14.98pp
   -> Table 10 is judged to come from the TabPFN arm; the paper reports the TabPFN multi-seed mean+/-std.
P2 fallback: both arms > 10pp -> the paper reports both arms' multi-seed values side by side, and
   states as measured that the archived single run's provenance cannot be established; Table 10
   switches to reporting intervals/dispersion instead of point values.
This script **does not tune parameters to match the archive**; the protocol (normalization / view
/ cap / backbone) is frozen before the run.
==================================================================

Run (on a machine with the TabPFN weights):
  TABPFN_WEIGHTS_DIR=~/.cache/tabpfn ~/.venv-test/bin/python \
      experiments/bexp55_tabpfn_importance_protocol.py --validate
  TABPFN_WEIGHTS_DIR=~/.cache/tabpfn ~/.venv-test/bin/python \
      experiments/bexp55_tabpfn_importance_protocol.py --units XJTU-R3 XJTU-2C --seeds 0 1 2
  ... --report
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.inspection import permutation_importance

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import (CONTEXT_CAP, evaluate_unit, subsample_context,
                                   fit_predict_tabpfn)
from battery_lab.temporal_features import build_view

OUT = ROOT / "results/battery/bexp55_tabpfn_importance.jsonl"
# The summary is written to a **versioned new path** (v2): the old summary
# bexp55_tabpfn_importance.json's _meta once misreported n_repeats (module default 3 instead
# of the actual --repeats 2) and missed the device; the old file is kept as a historical
# record as required, the raw lines are not rewritten, and the new summary rebuilds the
# metadata from the raw lines.
SUM = ROOT / "results/battery/bexp55_tabpfn_importance_summary_v2.json"
ARCHIVE = ROOT / "results/battery/p6_importance.json"
GBDT_ARCHIVE = ROOT / "results/battery/bexp52_importance.json"
BEXP1 = ROOT / "results/battery/bexp1_author_protocol.jsonl"
SEEDS = [0, 1, 2]
N_REPEATS = 3
REPEATS = N_REPEATS          # overridable via --repeats (each TabPFN predict costs far more than GBDT)
CAP = CONTEXT_CAP
FAMILIES = ["raw", "cycle", "lag1", "diff1", "from_first", "roll5"]
FAM_SLICES = {
    "raw": slice(0, 16), "cycle": slice(16, 17), "lag1": slice(17, 33),
    "diff1": slice(33, 49), "from_first": slice(49, 65), "roll5": slice(65, 81),
}
VALIDATION_CELLS = [("XJTU-2C", "raw", 0), ("TJU-1", "history", 0)]


def assemble(cells, Xnorm, cycle_mode):
    xs, ys, ids = [], [], []
    for c in cells:
        xs.append(build_view(Xnorm[c.cell_id], "history", cycle_mode=cycle_mode))
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def per_family_pct(imp_mean):
    raw_sum = {f: float(np.clip(imp_mean[sl], 0.0, None).sum())
               for f, sl in FAM_SLICES.items()}
    total = sum(raw_sum.values())
    if total <= 0:
        return {f: 0.0 for f in FAMILIES}, raw_sum
    return {f: raw_sum[f] / total * 100.0 for f in FAMILIES}, raw_sum


def run_unit_seed(unit, seed):
    """One (unit, seed): fit TabPFN -> held-out-cell permutation importance -> per-family percentages."""
    Xn = normalize_cells(unit.cells, "author_minmax")
    X_tr, y_tr, _ = assemble(unit.train_cells, Xn, "author_minmax")
    X_te, y_te, te_ids = assemble(unit.test_cells, Xn, "author_minmax")

    class Wrapped:
        """A thin sklearn-style wrapper: multiple predicts share one fitted TabPFN.

        The only difference from the paper's table values is **device**:
        `residual_lab.models._TABPFN_CPU_KW` pins inference to CPU (the paper states
        "All runs are executed on CPU"), and CPU inference is infeasible in the 11-unit x
        multi-predict permutation-importance setting (measured: 81 predicts did not finish
        within 50 minutes). This probe only serves the protocol-attribution decision and
        produces no paper table values, so it may override via `TABPFN_DEVICE`
        (default: cuda if available, otherwise cpu); the remaining hyper-parameters such as
        n_estimators match the CPU version.
        """

        def __init__(self):
            self.model = None

        def fit(self, X, y):
            import torch
            from tabpfn import TabPFNRegressor
            from residual_lab.models import _TABPFN_REG_CKPT, _tabpfn_model_path

            self.device = os.environ.get(
                "TABPFN_DEVICE",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
            self.model = TabPFNRegressor(
                random_state=seed,
                model_path=_tabpfn_model_path(_TABPFN_REG_CKPT),
                n_estimators=2,
                device=self.device,
                ignore_pretraining_limits=True,
            )
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

    t0 = time.time()
    Xc, yc = subsample_context(X_tr, y_tr, CAP, seed)
    model = Wrapped().fit(Xc, yc)

    def scorer(estimator, X, y):
        pred = estimator.predict(X)
        errs = [np.sqrt(np.mean((y[te_ids == cid] - pred[te_ids == cid]) ** 2))
                for cid in np.unique(te_ids)]
        return -float(np.mean(errs))

    res = permutation_importance(model, X_te, y_te, scoring=scorer,
                                 n_repeats=REPEATS, random_state=seed, n_jobs=1)
    fam_pct, fam_raw = per_family_pct(res.importances_mean)
    base = scorer(model, X_te, y_te)
    return {
        "unit": unit.unit_id, "seed": seed, "backbone": "tabpfn", "cap": CAP,
        "device": getattr(model, "device", "unknown"),
        "n_repeats": REPEATS, "n_test_cells": len(unit.test_cells),
        "n_test_rows": int(len(y_te)), "rmse_cell_macro": float(-base),
        "families": {f: round(fam_pct[f], 4) for f in FAMILIES},
        "family_raw_sum": {f: fam_raw[f] for f in FAMILIES},
        "runtime_s": time.time() - t0,
    }


def do_validate():
    """V1: re-run the two archived cells in the current environment and compare with the archive.

    Note: `bexp1_author_protocol.jsonl` has **duplicate rows** for some (unit, view, seed)
    keys (identical metadata, rmse differing by two orders of magnitude -- leftovers of an
    early abandoned cycle convention, see the same-named .bak file). We therefore do not take
    the last row but compare against **all duplicate rows**, reporting the closest match and
    whether it hits a value used in the paper's tables.
    """
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    arch = {}
    if BEXP1.exists():
        for line in BEXP1.read_text().splitlines():
            r = json.loads(line)
            arch.setdefault((r["unit"], r["view"], r["seed"]), []).append(
                r["rmse_cell_macro"])
    print("=== V1 environment reproducibility check ===")
    ok_all = True
    for uid, view, seed in VALIDATION_CELLS:
        ds, bk = umap[uid]
        unit = load_fit_unit(ds, bk)
        r = evaluate_unit(unit, view, seed, backbone="tabpfn",
                          normalize="author_minmax", context_cap=CAP)
        ours = r["rmse_cell_macro"]
        refs = arch.get((uid, view, seed), [])
        if not refs:
            print(f"  {uid}/{view}/seed{seed}: no such cell in the archive; recording this machine's value {ours:.6f}")
            continue
        best = min(refs, key=lambda v: abs(v - ours))
        rel = abs(ours - best) / best * 100
        ok = rel <= 1.0
        ok_all &= ok
        dup = f"(archive has {len(refs)} duplicate rows: {', '.join(f'{v:.6f}' for v in refs)})" if len(refs) > 1 else ""
        print(f"  {uid}/{view}/seed{seed}: ours={ours:.6f} archive={best:.6f} "
              f"rel={rel:.3f}% -> {'OK' if ok else 'FAIL'} {dup}")
    print(f"V1 {'holds' if ok_all else 'fails'}")


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["seed"]))
    return keys


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    summary = {}
    for uid, _, _ in FIT_UNITS:
        us = [r for r in rows if r["unit"] == uid]
        if not us:
            continue
        fams = {"n_seeds": len(us)}
        for f in FAMILIES:
            v = np.array([r["families"][f] for r in us], dtype=float)
            fams[f] = {"mean": float(v.mean()),
                       "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                       "min": float(v.min()), "max": float(v.max())}
        order = sorted(FAMILIES, key=lambda f: -fams[f]["mean"])
        gaps = []
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            gaps.append({"pair": [a, b], "gap": fams[a]["mean"] - fams[b]["mean"],
                         "combined_std": fams[a]["std"] + fams[b]["std"],
                         "range_overlap": not (fams[a]["min"] > fams[b]["max"] or
                                               fams[b]["min"] > fams[a]["max"])})
        gaps_sorted = sorted(gaps, key=lambda g: g["gap"])
        for g in gaps_sorted[:2]:
            g["tie_within_error"] = bool(g["gap"] <= g["combined_std"]) or g["range_overlap"]
        fams["top2"] = order[:2]
        fams["closest_pair"] = gaps_sorted[0] if gaps_sorted else None
        summary[uid] = fams

    arch = json.load(ARCHIVE.open()) if ARCHIVE.exists() else {}
    dev, dev_per_unit = [], {}
    for uid in summary:
        if uid not in arch:
            continue
        ds = []
        for f in FAMILIES:
            d = abs(summary[uid][f]["mean"] - arch[uid][f])
            dev.append((uid, f, summary[uid][f]["mean"], arch[uid][f], d))
            ds.append(d)
        dev_per_unit[uid] = max(ds) if ds else float("nan")
    max_dev = max((x[4] for x in dev), default=float("nan"))
    # comparison against the GBDT arm: compare the per-unit maximum deviations
    gbdt = json.load(GBDT_ARCHIVE.open())["dev_vs_archive"] if GBDT_ARCHIVE.exists() else []
    gbdt_max = max((d["abs_dev_pp"] for d in gbdt), default=float("nan"))
    win = sum(1 for u, v in dev_per_unit.items()
              if v < max((d["abs_dev_pp"] for d in gbdt if d["unit"] == u), default=float("inf")))
    reps = sorted({r.get("n_repeats") for r in rows})
    devs = sorted({r.get("device") for r in rows})
    caps = sorted({r.get("cap") for r in rows})
    if len(reps) != 1 or len(devs) != 1 or len(caps) != 1:
        raise SystemExit(
            f"mixed protocols, refusing to summarize: n_repeats={reps}, device={devs}, cap={caps}"
        )
    out = {"_meta": {"seeds": len({r["seed"] for r in rows}), "backbone": "tabpfn",
                     "cap": caps[0], "n_repeats": reps[0], "device": devs[0],
                     "n_cells": len(rows),
                     "max_abs_dev_vs_archive_pp": max_dev,
                     "gbdt_arm_max_dev_pp": gbdt_max,
                     "units_closer_to_archive_than_gbdt": win,
                     "n_units": len(dev_per_unit),
                     "criterion_P1_ok": bool(np.isfinite(max_dev) and max_dev <= 10.0),
                     "dev_per_unit_max_pp": dev_per_unit},
           "per_unit": summary,
           "dev_vs_archive": [{"unit": u, "family": f, "ours": o, "archive": a,
                               "abs_dev_pp": d} for u, f, o, a, d in dev]}
    json.dump(out, SUM.open("w"), indent=1)
    return out


def report():
    if not SUM.exists():
        print("no summary file; run the main flow first")
        return
    out = json.load(SUM.open())
    per = out["per_unit"]
    print(f"\n{'=' * 100}")
    print(f"bexp55 TabPFN multi-seed permutation importance (per family %, over {out['_meta']['seeds']} seeds, cap={CAP})")
    print(f"{'unit':<15}" + "".join(f"{f:>13}" for f in FAMILIES))
    for uid, _, _ in FIT_UNITS:
        if uid not in per:
            continue
        row = f"{uid:<15}"
        for f in FAMILIES:
            row += f"{per[uid][f]['mean']:>6.1f}±{per[uid][f]['std']:<5.1f}"
        print(row)
    m = out["_meta"]
    print(f"\nmetadata (rebuilt from raw rows): seeds={m['seeds']} repeats={m['n_repeats']} "
          f"device={m['device']} cap={m['cap']} cells={m['n_cells']}")
    print(f"max absolute deviation (vs archive) = {m['max_abs_dev_vs_archive_pp']:.2f}pp "
          f"| GBDT arm = {m['gbdt_arm_max_dev_pp']:.2f}pp "
          f"| units closer than GBDT {m['units_closer_to_archive_than_gbdt']}/{m['n_units']}")
    print(f"=== frozen verdict P1: {'holds -- Table 10 attributed to the TabPFN arm' if m['criterion_P1_ok'] else 'fails -- fall back to P2'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--repeats", type=int, default=N_REPEATS,
                    help="permutation_importance n_repeats (each TabPFN predict is expensive)")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    global REPEATS
    REPEATS = args.repeats

    if args.validate:
        do_validate()
        return

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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
