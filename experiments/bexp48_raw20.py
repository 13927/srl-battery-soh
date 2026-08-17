"""bexp48: the raw baseline needed for the 20-seed three-arm table (GBDT,
reference protocol, 20 seeds).

Table 9 used to show the 5-seed point estimates of bexp26, which did not agree
with the 8/11 win count from bexp44 (20 seeds) quoted in the text; the table on
its own implied 9/11. This script adds the 20-seed raw arm so that all three
columns of Table 9 can be recomputed at 20 seeds:
  Treatment(self) = bexp44 {u}|self;  Mispaired = bexp44 {u}|shuffled;
  the instrument arm is exactly 0.0 (affine invariance is analytic, so it needs
  no seeds).
The protocol matches bexp44 exactly (author_minmax + cycle_author_minmax; the raw
arm of the three is simply the version without the offset columns).
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, fit_predict_gbdt, subsample_context
from battery_lab.temporal_features import cycle_author_minmax

OUT = ROOT / "results/battery/bexp48_raw20.json"
SEEDS = list(range(20))

def run_unit(unit):
    Xn = normalize_cells(unit.cells, "author_minmax")
    def build(cells):
        xs, ys = [], []
        for c in cells:
            Z = Xn[c.cell_id]
            xs.append(np.hstack([Z, cycle_author_minmax(len(Z))]))
            ys.append(c.y)
        return np.vstack(xs), np.concatenate(ys)
    X_tr, y_tr = build(unit.train_cells)
    X_te, y_te = build(unit.test_cells)
    vals = []
    for s in SEEDS:
        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, s)
        vals.append(float(np.sqrt(np.mean((y_te - fit_predict_gbdt(Xc, yc, X_te, s)) ** 2))))
    return {"rmse": float(np.mean(vals)), "seed_vals": vals}

state = json.load(OUT.open()) if OUT.exists() else {}
umap = {u: (d, b) for u, d, b in FIT_UNITS}
for uid in [u for u, _, _ in FIT_UNITS]:
    if uid in state:
        continue
    t0 = time.time()
    state[uid] = run_unit(load_fit_unit(*umap[uid]))
    json.dump(state, OUT.open("w"), indent=1)
    print(f"{uid:<16} raw20={state[uid]['rmse']:.5f} [{time.time()-t0:.0f}s]", flush=True)
print("done")
