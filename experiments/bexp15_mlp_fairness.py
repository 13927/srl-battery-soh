"""bexp15: MLP fairness -- testing the "capacity for high-dimensional redundancy"
account.

Background (after the phase-1 upgrade):
  the original account (ICL benefits, gradient training is harmed) was refuted by
  GBDT (a gradient-boosted model that nonetheless benefits in 9/11). The new
  account: models that can handle high-dimensional redundant features benefit,
  those that cannot are harmed. The MLP here uses alpha=1e-4 (almost no
  regularisation) and a fixed (128,64), so it must overfit in the 81-dimensional
  history stack.

================== frozen predictions (committed before the run) ==================
M1: after source-side selection, the harm to the MLP from history narrows sharply
    -- the median harm over the three units falls from >300% to < 100%
M2: the direction is unchanged -- history still hurts the MLP (delta% > 0 in all
    three units)
M3: the selected alpha has median > 1e-4 (the search did pick stronger
    regularisation)

Criteria:
  M1 and M3 hold -> the new account is supported, and the interaction is recast as
                    a capacity/regularisation issue
  M1 fails (harm does not narrow) -> the harm is not a regularisation issue and
                    needs another explanation
  M2 fails (direction reverses)   -> the interaction claim must be rewritten
===========================================================

No leakage: the hyper-parameter search runs in 3 folds inside the source cells
(grouped by cell, GroupKFold); test cells never take part. Reference protocol,
comparable to bexp1.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, assemble_matrix, subsample_context

OUT = Path("results/battery/bexp15_mlp_fairness.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

UNITS = ["TJU-1", "HUST", "MIT"]        # big win / loss / neutral
SEEDS = [0, 1, 2]
ALPHAS = [1e-4, 1e-2, 1.0, 10.0]
HIDDENS = [(64, 32), (128, 64)]
# harm to the untuned MLP from bexp1 (overall RMSE)
BASELINE_HARM = {"TJU-1": 314.6, "HUST": 369.7, "MIT": 354.4}


def _mlp(alpha, hidden, seed):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=500,
                     early_stopping=True, random_state=seed),
    )


def select_hparams(X, y, groups, seed):
    """Select in 3 folds grouped by cell inside the source (test cells untouched)."""
    from sklearn.model_selection import GroupKFold

    n_splits = min(3, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    best, best_rmse = None, np.inf
    for alpha in ALPHAS:
        for hidden in HIDDENS:
            errs = []
            for tr, va in gkf.split(X, y, groups):
                m = _mlp(alpha, hidden, seed)
                m.fit(X[tr], y[tr])
                errs.append(np.sqrt(np.mean((y[va] - m.predict(X[va])) ** 2)))
            rmse = float(np.mean(errs))
            if rmse < best_rmse:
                best_rmse, best = rmse, (alpha, hidden)
    return best, best_rmse


def run(unit, view, seed):
    tr_cells, te_cells = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr_cells + te_cells, "author_minmax")
    X_tr, y_tr, g_tr = assemble_matrix(tr_cells, Xn, view, "author_minmax")
    X_te, y_te, te_ids = assemble_matrix(te_cells, Xn, view, "author_minmax")
    X_tr, y_tr_s = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
    # subsample_context returns no indices, so the same RNG reproduces the grouping
    if len(g_tr) > CONTEXT_CAP:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(g_tr), CONTEXT_CAP, replace=False)
        g_sub = g_tr[idx]
    else:
        g_sub = g_tr
    y_tr = y_tr_s

    (alpha, hidden), cv_rmse = select_hparams(X_tr, y_tr, g_sub, seed)
    m = _mlp(alpha, hidden, seed)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    per_cell = {c: float(np.sqrt(np.mean((y_te[te_ids == c]
                                        - pred[te_ids == c]) ** 2)))
                for c in np.unique(te_ids)}
    return {
        "unit": unit.unit_id, "view": view, "seed": seed,
        "alpha": alpha, "hidden": list(hidden), "cv_rmse": cv_rmse,
        "n_feat": int(X_te.shape[1]),
        "rmse_overall": float(np.sqrt(np.mean((y_te - pred) ** 2))),
        "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
    }


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["view"], r["seed"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=UNITS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        ds, bk = umap[uid]
        unit = load_fit_unit(ds, bk)
        for view in ("raw", "history"):
            for seed in args.seeds:
                if (uid, view, seed) in done:
                    continue
                t0 = time.time()
                r = run(unit, view, seed)
                with OUT.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"{uid:<8}{view:<8}seed={seed} alpha={r['alpha']:<7} "
                      f"hidden={tuple(r['hidden'])} "
                      f"overall={r['rmse_overall']:.6f} [{time.time()-t0:.0f}s]",
                      flush=True)
    report()


def report():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    print(f"\n{'='*80}\nhistory effect on the tuned MLP "
          f"(overall RMSE, mean over seeds)")
    print(f"{'unit':<9}{'raw':>10}{'history':>10}{'tuned d%':>10}"
          f"{'untuned d%':>11}{'narrowed':>9}  chosen (alpha,hidden)")
    harms, alphas_sel = [], []
    for uid in UNITS:
        m = {}
        for v in ("raw", "history"):
            x = [r["rmse_overall"] for r in rows if r["unit"] == uid and r["view"] == v]
            if x:
                m[v] = float(np.mean(x))
        if len(m) < 2:
            continue
        d = (m["history"] - m["raw"]) / m["raw"] * 100
        base = BASELINE_HARM[uid]
        harms.append(d)
        sel = [(r["alpha"], tuple(r["hidden"])) for r in rows
               if r["unit"] == uid and r["view"] == "history"]
        alphas_sel += [a for a, _ in sel]
        print(f"{uid:<9}{m['raw']:>10.6f}{m['history']:>10.6f}{d:>+10.1f}"
              f"{base:>+10.1f}{f'{(1-d/base)*100:.0f}%':>9}  {sel}")

    if not harms:
        return
    med = float(np.median(harms))
    med_alpha = float(np.median(alphas_sel)) if alphas_sel else np.nan
    m1 = med < 100
    m2 = all(h > 0 for h in harms)
    m3 = med_alpha > 1e-4
    print(f"\n  === frozen verdict ===")
    print(f"  M1 (median harm < +100%): {med:+.1f}% -> "
          + ("holds" if m1 else "falsified")
          + f"   (untuned median +{np.median(list(BASELINE_HARM.values())):.1f}%)")
    print(f"  M2 (direction unchanged, all >0): {[f'{h:+.1f}' for h in harms]} -> "
          + ("holds" if m2 else "falsified"))
    print(f"  M3 (selected alpha median > 1e-4): {med_alpha:g} -> "
          + ("holds" if m3 else "falsified"))
    print("\n  the capacity-for-redundancy account: "
          + ("supported" if (m1 and m3) else "not supported"))
    print("  interaction direction: "
          + ("preserved (history still hurts the MLP)" if m2
             else "reversed, the claim must be rewritten"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
