"""bexp29: closing out the construction-selection line. The pre-registered
selection rule (commit 48720c3) picked history (81 dimensions) on the development
set, i.e. the incumbent representation, so the prospective check collapses into
the bexp13 baseline (already verified three times) and no new criterion remains
to test.

This script is therefore an observation only: ff1/ff5/history on MATR-CLO
(3 seeds, no criterion), recording whether the HUST repair effect reappears on
the unexposed library.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from bexp5_unexposed_frozen import (CONTEXT_CAP, assemble as b5_assemble,
                                    load_unit_npz, prepare, subsample_context)
from bexp28_rsr import from_first_k
from residual_lab.models import make_model

OUT = Path("results/battery/bexp29_rsr_observe.json")
unit = load_unit_npz()
src, tgt = unit.train_cells, unit.test_cells
normed, ind = prepare(unit, src, tgt)

def build(cells, cond):
    xs, ys = [], []
    for c in cells:
        Z = normed[c.cell_id]
        from battery_lab.temporal_features import cycle_scaled, add_history_features
        cols = [Z, cycle_scaled(len(Z))]
        if cond == "ff1": cols.append(from_first_k(Z, 1))
        elif cond == "ff5": cols.append(from_first_k(Z, 5))
        elif cond == "history": cols.append(add_history_features(Z))
        ind_col = ind[c.cell_id].mean(axis=1, keepdims=True)
        cols.append(ind_col)
        xs.append(np.hstack(cols)); ys.append(c.y)
    return np.vstack(xs), np.concatenate(ys)

res = {}
for cond in ("ff1", "ff5", "history"):
    X_tr, y_tr = build(src, cond); X_te, y_te = build(tgt, cond)
    vals = []
    for seed in (0, 1, 2):
        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
        m = make_model("tabpfn_reg", seed=seed); m.fit(Xc, yc)
        vals.append(float(np.sqrt(np.mean((y_te - m.predict(X_te)) ** 2))))
    res[cond] = {"rmse": float(np.mean(vals)), "seeds": vals}
    print(cond, res[cond], flush=True)
json.dump(res, OUT.open("w"), indent=1)
