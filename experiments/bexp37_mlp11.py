"""bexp37: the fairly tuned MLP extended to all 11 units.

Protocol: the reference protocol (author_minmax + cycle_author_minmax), an alpha
grid of {1e-4, 1e-3, 1e-2, 1e-1, 1}, alpha chosen per seed on an 80/20 split of
the training rows (validation RMSE), then refitted on all training rows.
Conditions: {raw, history} x 11 units x 3 seeds, context cap 2048.

================ frozen criteria (committed before the run) ================
T1: history is no worse than raw on >= 7 of 11 units -> the claim of coverage
    across three model families holds
=======================================================
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, subsample_context
from battery_lab.temporal_features import add_history_features, cycle_author_minmax

OUT = ROOT / "results/battery/bexp37_mlp11.json"
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

def build(cells, Xn, cond):
    xs, ys = [], []
    for c in cells:
        Z = Xn[c.cell_id]; cols = [Z, cycle_author_minmax(len(Z))]
        if cond == "history": cols.append(add_history_features(Z))
        xs.append(np.hstack(cols)); ys.append(c.y)
    return np.vstack(xs), np.concatenate(ys)

state = json.load(OUT.open()) if OUT.exists() else {}
umap = {u:(d,b) for u,d,b in FIT_UNITS}
for uid,_,_ in FIT_UNITS:
    unit = load_fit_unit(*umap[uid])
    Xn = normalize_cells(unit.cells, "author_minmax")
    for cond in ("raw","history"):
        key = f"{uid}|{cond}"
        if key in state: continue
        X_tr0, y_tr0 = build(unit.train_cells, Xn, cond)
        X_te, y_te = build(unit.test_cells, Xn, cond)
        t0 = time.time(); vals = []
        for seed in (0,1,2):
            Xc, yc = subsample_context(X_tr0, y_tr0, CONTEXT_CAP, seed)
            sc = StandardScaler().fit(Xc)
            Xs = sc.transform(Xc); Xts = sc.transform(X_te)
            rng = np.random.default_rng(seed); idx = rng.permutation(len(Xs))
            ntr = int(len(Xs)*0.8); itr, iva = idx[:ntr], idx[ntr:]
            best_a, best_v = None, 9e9
            for a in ALPHAS:
                m = MLPRegressor(hidden_layer_sizes=(64,64), alpha=a, max_iter=400, random_state=seed,
                                 early_stopping=True).fit(Xs[itr], yc[itr])
                v = float(np.sqrt(np.mean((yc[iva]-m.predict(Xs[iva]))**2)))
                if v < best_v: best_a, best_v = a, v
            m = MLPRegressor(hidden_layer_sizes=(64,64), alpha=best_a, max_iter=400, random_state=seed,
                             early_stopping=True).fit(Xs, yc)
            vals.append(float(np.sqrt(np.mean((y_te-m.predict(Xts))**2))))
        state[key] = {"rmse": float(np.mean(vals)), "seed_vals": vals}
        json.dump(state, OUT.open("w"), indent=1)
        print(f"{key:<24} rmse={state[key]['rmse']:.5f} [{time.time()-t0:.0f}s]", flush=True)

n = sum(1 for u,_,_ in FIT_UNITS
        if state[f"{u}|history"]["rmse"] <= state[f"{u}|raw"]["rmse"]*1.0001)
print(f"\nT1 tuned MLP, history no worse than raw: {n}/11 "
      f"(criterion >=7) -> {'holds' if n >= 7 else 'falsified'}")
