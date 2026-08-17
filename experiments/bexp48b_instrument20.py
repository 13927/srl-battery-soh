"""bexp48b: the instrument arm at 20 seeds.

bexp48 covered the raw arm only. The instrument arm (the mean first cycle of the
source cells, an affine negative control) measured 0.00 per cent at 5 seeds in
bexp26, while Table 9 now states 20 seeds. This script measures the instrument
arm at 20 seeds, removing the mismatch between a 20-seed header and a 5-seed
column. Affine invariance makes zero the theoretical value for tree models; what
is verified here is the empirical fact that the measurement is indeed zero.
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, fit_predict_gbdt, subsample_context
from battery_lab.temporal_features import cycle_author_minmax

OUT = ROOT / "results/battery/bexp48b_instrument20.json"
SEEDS = list(range(20))

def run_unit(unit):
    Xn = normalize_cells(unit.cells, "author_minmax")
    # instrument: the anchor is the per-column mean first cycle of the source
    # (training) cells, i.e. one constant vector for every cell
    train_anchor = np.mean([Xn[c.cell_id][:1] for c in unit.train_cells], axis=0)
    def build(cells, offset):
        xs, ys = [], []
        for c in cells:
            Z = Xn[c.cell_id]
            base = np.hstack([Z, cycle_author_minmax(len(Z))])
            xs.append(np.hstack([base, Z - train_anchor]) if offset else base)
            ys.append(c.y)
        return np.vstack(xs), np.concatenate(ys)
    Xr_tr, y_tr = build(unit.train_cells, False)
    Xr_te, y_te = build(unit.test_cells, False)
    Xi_tr, _ = build(unit.train_cells, True)
    Xi_te, _ = build(unit.test_cells, True)
    raw_vals, inst_vals = [], []
    for s in SEEDS:
        Xc, yc = subsample_context(Xr_tr, y_tr, CONTEXT_CAP, s)
        idx_pred = fit_predict_gbdt(Xc, yc, Xr_te, s)
        raw_vals.append(float(np.sqrt(np.mean((y_te - idx_pred) ** 2))))
        Xci, _ = subsample_context(Xi_tr, y_tr, CONTEXT_CAP, s)
        inst_vals.append(float(np.sqrt(np.mean((y_te - fit_predict_gbdt(Xci, yc, Xi_te, s)) ** 2))))
    raw, inst = float(np.mean(raw_vals)), float(np.mean(inst_vals))
    return {"raw": raw, "instrument": inst,
            "delta_pct": (inst - raw) / raw * 100}

state = json.load(OUT.open()) if OUT.exists() else {}
umap = {u: (d, b) for u, d, b in FIT_UNITS}
for uid in [u for u, _, _ in FIT_UNITS]:
    if uid in state:
        continue
    t0 = time.time()
    state[uid] = run_unit(load_fit_unit(*umap[uid]))
    json.dump(state, OUT.open("w"), indent=1)
    print(f"{uid:<16} raw={state[uid]['raw']:.5f} inst={state[uid]['instrument']:.5f} "
          f"Δ={state[uid]['delta_pct']:+.3f}% [{time.time()-t0:.0f}s]", flush=True)
print("done")
