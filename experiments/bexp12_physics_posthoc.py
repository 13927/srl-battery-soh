"""bexp12: borrowing from PINN/QPINN -- training-free physics priors.

A. trajectory post-processing: one TabPFN inference reused across 6 variants
   (cost ~ 0)
B. Nystroem kernel feature expansion (the classical counterpart of the QPINN
   quantum feature map)

Benchmark: the official PINN4SOH results (external/PINN4SOH/results analysis/
processed results/Ours-*.xlsx, overall RMSE averaged over 10 repetitions).

The reference protocol is used throughout (as bexp1) to keep the comparison
fair.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.physics_posthoc import POSTPROC, apply_postproc, nystroem_expand
from battery_lab.protocols import CONTEXT_CAP, subsample_context
from battery_lab.temporal_features import build_view

OUT = Path("results/battery/bexp12_physics_posthoc.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]

# official PINN4SOH overall RMSE (mean of 10 repetitions, read from the reference
# xlsx)
PINN_OFFICIAL = {"XJTU": 0.009410, "TJU": 0.015789,
                 "HUST": 0.008685, "MIT": 0.007400}
LIB_OF = {**{f"XJTU-{b}": "XJTU" for b in
             ["2C", "3C", "R2.5", "R3", "RW", "satellite"]},
          **{f"TJU-{i}": "TJU" for i in (1, 2, 3)},
          "HUST": "HUST", "MIT": "MIT"}


def _stack(cells, Xn, view, cycle_mode):
    xs, ys, cids = [], [], []
    for c in cells:
        xs.append(build_view(Xn[c.cell_id], view, cycle_mode=cycle_mode))
        ys.append(c.y)
        cids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(cids)


def run_unit(unit, view, seed, expand=0):
    """One inference, then every post-processing variant. With expand>0 the
    Nystroem expansion is applied first."""
    from residual_lab.models import make_model

    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "author_minmax")
    X_tr, y_tr, _ = _stack(tr, Xn, view, "author_minmax")
    X_te, y_te, cid_te = _stack(te, Xn, view, "author_minmax")
    if expand:
        X_tr, X_te = nystroem_expand(X_tr, X_te, n_components=expand, seed=seed)
    X_ctx, y_ctx = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)

    m = make_model("tabpfn_reg", seed=seed)
    m.fit(X_ctx, y_ctx)
    raw_pred = m.predict(X_te)

    rows = []
    for mode in POSTPROC:
        p = apply_postproc(raw_pred, cid_te, mode)
        per_cell = {c: float(np.sqrt(np.mean((y_te[cid_te == c]
                                             - p[cid_te == c]) ** 2)))
                    for c in np.unique(cid_te)}
        rows.append({
            "unit": unit.unit_id, "view": view, "seed": seed,
            "postproc": mode, "expand": expand, "n_feat": int(X_te.shape[1]),
            "rmse_overall": float(np.sqrt(np.mean((y_te - p) ** 2))),
            "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
            "mae_overall": float(np.mean(np.abs(y_te - p))),
            "bias": float(np.mean(p - y_te)),      # positive = overestimate
                                                  # (isolates bias correction)
            "resid_std": float(np.std(p - y_te)),   # variance component
        })
    return rows


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["view"], r["seed"], r["expand"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--views", nargs="+", default=["history"])
    ap.add_argument("--expand", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        ds, bk = umap[uid]
        unit = load_fit_unit(ds, bk)
        for view in args.views:
            for ex in args.expand:
                for seed in args.seeds:
                    if (uid, view, seed, ex) in done:
                        continue
                    t0 = time.time()
                    rows = run_unit(unit, view, seed, expand=ex)
                    with OUT.open("a") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    base = [r for r in rows if r["postproc"] == "none"][0]
                    best = min(rows, key=lambda r: r["rmse_overall"])
                    print(f"{uid:<15}{view:<8}ex={ex:<4}seed={seed} "
                          f"none={base['rmse_overall']:.6f} "
                          f"best={best['postproc']}:{best['rmse_overall']:.6f} "
                          f"({(best['rmse_overall']-base['rmse_overall'])/base['rmse_overall']*100:+.1f}%) "
                          f"[{time.time()-t0:.0f}s]", flush=True)
    report()


def report():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    modes = list(POSTPROC)
    for ex in sorted({r["expand"] for r in rows}):
        sub = [r for r in rows if r["expand"] == ex]
        print(f"\n{'='*84}\npost-processing (expand={ex}, overall RMSE, mean "
              f"over seeds)")
        print(f"{'unit':<15}" + "".join(f"{m[:10]:>11}" for m in modes))
        for uid in [u for u, _, _ in FIT_UNITS]:
            u = [r for r in sub if r["unit"] == uid]
            if not u:
                continue
            line = f"{uid:<15}"
            for m in modes:
                v = [r["rmse_overall"] for r in u if r["postproc"] == m]
                line += f"{np.mean(v):>11.6f}" if v else f"{'-':>11}"
            print(line)

        # library-level aggregate vs the published PINN4SOH
        print(f"\nlibrary aggregate vs published PINN4SOH (overall RMSE)")
        print(f"{'library':<9}{'published':>10}"
              + "".join(f"{m[:10]:>11}" for m in modes))
        for lib, off in PINN_OFFICIAL.items():
            us = [u for u, l in LIB_OF.items() if l == lib]
            line = f"{lib:<7}{off:>10.6f}"
            for m in modes:
                vals = []
                for uid in us:
                    v = [r["rmse_overall"] for r in sub if r["unit"] == uid
                         and r["postproc"] == m]
                    if v:
                        vals.append(np.mean(v))
                line += f"{np.mean(vals):>11.6f}" if len(vals) == len(us) else f"{'-':>11}"
            print(line)
        print(f"\nrelative to published, delta% (negative = ours better)")
        print(f"{'library':<9}" + "".join(f"{m[:10]:>11}" for m in modes))
        for lib, off in PINN_OFFICIAL.items():
            us = [u for u, l in LIB_OF.items() if l == lib]
            line = f"{lib:<7}"
            for m in modes:
                vals = []
                for uid in us:
                    v = [r["rmse_overall"] for r in sub if r["unit"] == uid
                         and r["postproc"] == m]
                    if v:
                        vals.append(np.mean(v))
                if len(vals) == len(us):
                    line += f"{(np.mean(vals)-off)/off*100:>+11.1f}"
                else:
                    line += f"{'-':>11}"
            print(line)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
