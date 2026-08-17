"""bexp41: separating protocol from substance on MATR-CLO -- was the "self arm
hurts" result of bexp40 an effect of the normalisation or a genuine
counter-example?

Background: the same MATR-CLO library gave two opposite readings:
  - bexp40 (server): GBDT + per-cell full-lifetime min-max -> the self arm hurt by
    +6.3 per cent and the pairing signal vanished
  - bexp29 (local):  TabPFN + source-side normalisation -> ff1 (0.0136) beat
    history (0.0145)
The carrier account predicts exactly this: per-cell full-lifetime min-max is
itself an implicit self-reference (an offline convention), and where it is present
an explicit from_first column is marginal or harmful (already seen twice, in
bexp31/33). This experiment isolates the normalisation as the only variable.

Design (data cleaning reproduces bexp40 exactly: intersection of valid columns
across cells, then per-cell removal of residual NaN rows):
  normalisation: percell   = per-cell full-lifetime min-max(-1,1)  <- as bexp40
                 source    = min-max fitted on the source pool, test rows only
                             transformed <- the strict, deployable form
                 sourcecyc = source + a cycle/200 column
  backbone:      gbdt (deterministic, all training rows, as bexp40); tabpfn
                 (optional, cap=2048, 3 seeds)
  three arms:    raw / self (+X_t - X_first_own) / shuffled
                 (+X_t - X_first_other, cyclic-shift derangement with seed
                 20260731)

================ frozen criteria (committed before the run) ================
V1a (replication): under percell x gbdt, the self arm is >= 0 relative to raw
     (the harm seen in bexp40 should reappear)
V1b (main criterion): under source x gbdt, the self arm is < 0 relative to raw
   -> holds: bexp40 is attributed to the normalisation protocol, the carrier
      account is confirmed a third time on an independent library, and the claim
      survives (its scope stated as "under source-side normalisation")
   -> fails: look at sourcecyc next; if that is also >= 0, MATR-CLO is a genuine
      counter-example, the claim narrows to the PINN4SOH benchmark family, and the
      paper turns towards auditing the external validity of representation claims
V2 (pairing signal): under source x gbdt, self < shuffled
The TabPFN arm is an observation only (the server tabpfn 8.2.0 and the local v3
are different generations and are not pooled with the local line)
============================================================

Run on the server:
  conda activate pfn_env_820
  python experiments/bexp41_matrclo_protocol.py                # GBDT, minutes
  python experiments/bexp41_matrclo_protocol.py --with-tabpfn  # add the TabPFN arm
  python experiments/bexp41_matrclo_protocol.py --report
Returns: results/battery/bexp41_matrclo_protocol.json + stdout
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

NPZ = Path("external/unexposed/matr_clo_unit.npz")
OUT = Path("results/battery/bexp41_matrclo_protocol.json")
MAP_SEED = 20260731
TAB_SEEDS = [0, 1, 2]
TAB_CAP = 2048


def load_cells():
    """Reproduce the bexp40 cleaning: intersection of valid columns (25), then
    per-cell removal of residual NaN rows."""
    z = np.load(NPZ, allow_pickle=False)
    ids = [str(x) for x in z["cell_ids"]]
    cells = []
    for cid in ids:
        cells.append({"id": cid, "X": z[f"X_{cid}"], "y": z[f"y_{cid}"],
                      "test": bool(z[f"t_{cid}"])})
    n_col = cells[0]["X"].shape[1]
    keep = [j for j in range(n_col)
            if all(not np.isnan(c["X"][:, j]).all() for c in cells)]
    for c in cells:
        Xk = c["X"][:, keep]
        ok = ~np.isnan(Xk).any(axis=1)
        c["X"], c["y"] = Xk[ok], c["y"][ok]
    print(f"cells={len(cells)} keep_cols={len(keep)} "
          f"train={sum(not c['test'] for c in cells)} "
          f"test={sum(c['test'] for c in cells)}", flush=True)
    return cells


def normalize(cells, mode):
    tr = [c for c in cells if not c["test"]]
    if mode == "percell":
        out = {}
        for c in cells:
            mn, mx = c["X"].min(0, keepdims=True), c["X"].max(0, keepdims=True)
            rng = np.where(mx - mn == 0, 1.0, mx - mn)
            out[c["id"]] = 2 * (c["X"] - mn) / rng - 1
        return out
    pool = np.vstack([c["X"] for c in tr])
    mn, mx = pool.min(0, keepdims=True), pool.max(0, keepdims=True)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return {c["id"]: 2 * (c["X"] - mn) / rng - 1 for c in cells}


def derangement(ids):
    rng = np.random.default_rng(MAP_SEED)
    perm = list(rng.permutation(ids))
    return {perm[i]: perm[(i + 1) % len(perm)] for i in range(len(perm))}


def assemble(cells, Xn, arm, norm_mode, ref_map):
    xs, ys = [], []
    for c in cells:
        Z = Xn[c["id"]]
        cols = [Z]
        if norm_mode == "sourcecyc":
            cols.append((np.arange(len(Z)) / 200.0).reshape(-1, 1))
        if arm == "self":
            cols.append(Z - Z[:1])
        elif arm == "shuffled":
            cols.append(Z - Xn[ref_map[c["id"]]][:1])
        xs.append(np.hstack(cols))
        ys.append(c["y"])
    return np.vstack(xs), np.concatenate(ys)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main(with_tabpfn=False):
    from sklearn.ensemble import HistGradientBoostingRegressor
    cells = load_cells()
    tr = [c for c in cells if not c["test"]]
    te = [c for c in cells if c["test"]]
    ref_map = derangement([c["id"] for c in cells])
    state = json.load(OUT.open()) if OUT.exists() else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for norm_mode in ("percell", "source", "sourcecyc"):
        Xn = normalize(cells, "percell" if norm_mode == "percell" else "source")
        for arm in ("raw", "self", "shuffled"):
            X_tr, y_tr = assemble(tr, Xn, arm, norm_mode, ref_map)
            X_te, y_te = assemble(te, Xn, arm, norm_mode, ref_map)
            key = f"gbdt|{norm_mode}|{arm}"
            if key not in state:
                t0 = time.time()
                m = HistGradientBoostingRegressor(random_state=0)
                m.fit(X_tr, y_tr)                      # as bexp40: all training rows
                state[key] = rmse(y_te, m.predict(X_te))
                json.dump(state, OUT.open("w"), indent=1)
                print(f"{key:<26} rmse={state[key]:.5f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            if with_tabpfn:
                key_t = f"tabpfn|{norm_mode}|{arm}"
                if key_t in state:
                    continue
                from tabpfn import TabPFNRegressor
                vals = []
                t0 = time.time()
                for seed in TAB_SEEDS:
                    rng = np.random.default_rng(seed)
                    idx = (rng.permutation(len(X_tr))[:TAB_CAP]
                           if len(X_tr) > TAB_CAP else np.arange(len(X_tr)))
                    mt = TabPFNRegressor(random_state=seed)
                    mt.fit(X_tr[idx], y_tr[idx])
                    vals.append(rmse(y_te, mt.predict(X_te)))
                state[key_t] = {"rmse": float(np.mean(vals)), "seed_vals": vals}
                json.dump(state, OUT.open("w"), indent=1)
                print(f"{key_t:<26} rmse={state[key_t]['rmse']:.5f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    print(f"\n{'='*72}\nMATR-CLO protocol separation "
          f"(delta% against the raw arm of the same cell; negative = better)")
    print(f"{'cell':<22}{'raw':>9}{'self':>9}{'shuf':>9}{'self delta%':>12}"
          f"{'self<shuf':>11}")
    get = lambda k: state[k] if isinstance(state.get(k), float) else \
        state.get(k, {}).get("rmse")
    for bk in ("gbdt", "tabpfn"):
        for nm in ("percell", "source", "sourcecyc"):
            r0, r1, r2 = (get(f"{bk}|{nm}|{a}") for a in ("raw", "self", "shuffled"))
            if r0 is None:
                continue
            d = (r1 - r0) / r0 * 100
            print(f"{bk}×{nm:<15}{r0:>9.5f}{r1:>9.5f}{r2:>9.5f}{d:>+9.1f}"
                  f"{'✓' if r1 < r2 else '✗':>10}")
    g = lambda nm, a: state.get(f"gbdt|{nm}|{a}")
    if all(g(nm, a) is not None for nm in ("percell", "source")
           for a in ("raw", "self", "shuffled")):
        v1a = g("percell", "self") >= g("percell", "raw")
        v1b = g("source", "self") < g("source", "raw")
        v2 = g("source", "self") < g("source", "shuffled")
        print("\nV1a replication (percell: the self arm hurts): "
              + ("yes" if v1a else "NOT replicated"))
        print("V1b main criterion (source: the self arm helps): "
              + ("holds -- a protocol effect, the claim survives" if v1b
                 else "fails -- check sourcecyc, or a genuine counter-example"))
        print("V2 pairing signal (source: self < shuffled): "
              + ("yes" if v2 else "no"))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        main(with_tabpfn="--with-tabpfn" in sys.argv)
