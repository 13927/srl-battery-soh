"""bexp42: the convention effect over all 11 units, and its correlation with
carrier importance.

Design: everything except the cycle column's convention is frozen (author_minmax
normalisation plus the full history stack), GBDT with 5 seeds. Convention A =
cycle_author_minmax (per-cell full lifetime, offline only); convention B =
cycle_scaled (/200, computable in service).

================ frozen criteria (committed before the run) ================
C1: the per-unit sensitivity delta% (B relative to A) correlates with that unit's
    permutation importance of the cycle column (p6_importance.json) at
    Spearman rho > 0.5 and p < 0.05
    -> holds: "the size of the convention effect is predicted by the carrier
       structure" may be stated as a formal claim
    -> fails: downgrade to a descriptive observation on three units and rewrite
       the section accordingly
=======================================================
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import spearmanr

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, fit_predict_gbdt, subsample_context
from battery_lab.temporal_features import (add_history_features,
                                           cycle_author_minmax, cycle_scaled)

OUT = ROOT / "results/battery/bexp42_cycle_convention.json"
IMP = ROOT / "results/battery/p6_importance.json"
SEEDS = [0, 1, 2, 3, 4]
MODES = {"lifetime": cycle_author_minmax, "deployable": cycle_scaled}


def run_unit(unit, cyc_fn):
    Xn = normalize_cells(unit.cells, "author_minmax")

    def build(cells):
        xs, ys = [], []
        for c in cells:
            Z = Xn[c.cell_id]
            xs.append(np.hstack([Z, cyc_fn(len(Z)), add_history_features(Z)]))
            ys.append(c.y)
        return np.vstack(xs), np.concatenate(ys)

    X_tr, y_tr = build(unit.train_cells)
    X_te, y_te = build(unit.test_cells)
    vals = []
    for s in SEEDS:
        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, s)
        vals.append(float(np.sqrt(np.mean((y_te - fit_predict_gbdt(Xc, yc, X_te, s)) ** 2))))
    return {"rmse": float(np.mean(vals)), "seed_vals": vals}


def main():
    state = json.load(OUT.open()) if OUT.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in [u for u, _, _ in FIT_UNITS]:
        todo = [m for m in MODES if f"{uid}|{m}" not in state]
        if not todo:
            continue
        unit = load_fit_unit(*umap[uid])
        t0 = time.time()
        for m in todo:
            state[f"{uid}|{m}"] = run_unit(unit, MODES[m])
            json.dump(state, OUT.open("w"), indent=1)
        a = state[f"{uid}|lifetime"]["rmse"]
        b = state[f"{uid}|deployable"]["rmse"]
        print(f"{uid:<16} lifetime={a:.5f} deployable={b:.5f} "
              f"Δ={(b-a)/a*100:+.1f}% [{time.time()-t0:.0f}s]", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    imp = json.load(IMP.open())
    print(f"\n{'='*72}\nconvention sensitivity vs cycle-carrier importance "
          f"(11 units)")
    print(f"{'unit':<16}{'lifetime':>10}{'deployable':>12}{'delta%':>9}"
          f"{'cycle imp%':>12}")
    ds, cs = [], []
    for uid, _, _ in FIT_UNITS:
        ka, kb = f"{uid}|lifetime", f"{uid}|deployable"
        if ka not in state or kb not in state:
            continue
        a, b = state[ka]["rmse"], state[kb]["rmse"]
        d = (b - a) / a * 100
        c = imp[uid]["cycle"]
        ds.append(d); cs.append(c)
        print(f"{uid:<16}{a:>10.5f}{b:>12.5f}{d:>+9.1f}{c:>12.1f}")
    r = spearmanr(cs, ds)
    ok = r.statistic > 0.5 and r.pvalue < 0.05
    print(f"\nSpearman(cycle importance, sensitivity) rho={r.statistic:+.3f} "
          f"p={r.pvalue:.4f}  (n={len(ds)})")
    print("C1 (rho>0.5 and p<0.05): "
          + ("holds -- the effect size is predicted by the carrier structure"
             if ok else "falsified -- downgraded to a descriptive observation"))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        main()
