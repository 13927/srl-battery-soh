"""bexp33: a 2x2 disambiguation -- the lift against feature richness x
normalisation convention.

Two competing explanations for the marginal lift in bexp31: either the implicit
baseline reference is already present in the per-cell normalisation (our reading),
or the lift only helps when features are sparse (the alternative). The 2x2
separates them:

  feature axis:       16d (PINN4SOH CSV) vs 67d (SOHbenchmark xlsx, same cells)
  normalisation axis: percell (per-cell full-lifetime min-max, an offline
                      convention) vs source (fitted on the source side, strict)

Each cell measures raw / raw+lift (GBDT, 5 seeds, six XJTU batches, official PINN
test cells 4/8[,14]). The lift is a within-cell contrast (the cycle column follows
the normalisation axis: percell -> author_minmax, source -> scaled200; it is the
same in both conditions and so does not affect the contrast).

================ frozen criteria (committed before the run) ================
D1 (key): in the 67d x source cell, the median lift is <= -3 per cent or it
          improves >= 4 of 6 batches
          -> holds: the "sparse features only" explanation is ruled out (with rich
             features, removing the implicit reference brings the lift back)
          -> fails: narrow the stated scope accordingly
D2 (replication): in the 67d x percell cell, |median lift| < 3 per cent (the
          marginal result of bexp31 should reappear)
=======================================================
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import CONTEXT_CAP, fit_predict_gbdt, subsample_context
from battery_lab.temporal_features import cycle_author_minmax, cycle_scaled

BENCH = ROOT / "external/SOHbenchmark/data/XJTU/handcraft_features"
OUT = ROOT / "results/battery/bexp33_2x2.json"
SEEDS = [0, 1, 2, 3, 4]
XJTU_UNITS = ["XJTU-2C", "XJTU-3C", "XJTU-R2.5", "XJTU-R3", "XJTU-RW",
              "XJTU-satellite"]
TEST_IDS = {1: [4, 8], 2: [4, 8, 14], 3: [4, 8], 4: [4, 8], 5: [4, 8], 6: [4, 8]}


def load_16d(unit_idx):
    """The 16d PINN4SOH features, returned as (train_raw, test_raw) under the
    official split."""
    uid = XJTU_UNITS[unit_idx]
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    unit = load_fit_unit(*umap[uid])
    tr = [(c.X.astype(np.float64), c.y) for c in unit.train_cells]
    te = [(c.X.astype(np.float64), c.y) for c in unit.test_cells]
    return tr, te


def load_67d(batch):
    """The 67d SOHbenchmark features (unnormalised), split by official test id."""
    dfs = pd.read_excel(BENCH / f"batch-{batch}_features.xlsx", sheet_name=None)
    cells = []
    for name, df in dfs.items():
        x = np.array(df.iloc[:, :-1], dtype=np.float64)
        y = np.array(df["label"], dtype=np.float64) / 2.0
        cells.append((x, y))
    te_idx = {i - 1 for i in TEST_IDS[batch]}
    tr = [c for i, c in enumerate(cells) if i not in te_idx]
    te = [c for i, c in enumerate(cells) if i in te_idx]
    return tr, te


def normalize(tr, te, mode):
    """percell: min-max(-1,1) over each cell's own lifetime; source: fitted on the
    source pool, then applied to everything."""
    if mode == "percell":
        def f(x):
            mn, mx = x.min(0, keepdims=True), x.max(0, keepdims=True)
            rng = np.where(mx - mn == 0, 1.0, mx - mn)
            return 2 * (x - mn) / rng - 1
        return [(f(x), y) for x, y in tr], [(f(x), y) for x, y in te]
    pool = np.vstack([x for x, _ in tr])
    mn, mx = pool.min(0, keepdims=True), pool.max(0, keepdims=True)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    g = lambda x: 2 * (x - mn) / rng - 1
    return [(g(x), y) for x, y in tr], [(g(x), y) for x, y in te]


def assemble(cells, cond, norm_mode):
    xs, ys = [], []
    cyc_fn = cycle_author_minmax if norm_mode == "percell" else cycle_scaled
    for Z, y in cells:
        cols = [Z, cyc_fn(len(Z))]
        if cond == "srl":
            cols.append(Z - Z[:1])
        xs.append(np.hstack(cols))
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def main():
    state = json.load(OUT.open()) if OUT.exists() else {}
    for b in range(1, 7):
        for feat in ("16d", "67d"):
            raw_tr, raw_te = (load_16d(b - 1) if feat == "16d" else load_67d(b))
            for norm in ("percell", "source"):
                tr, te = normalize(raw_tr, raw_te, norm)
                for cond in ("raw", "srl"):
                    key = f"b{b}|{feat}|{norm}|{cond}"
                    if key in state:
                        continue
                    X_tr, y_tr = assemble(tr, cond, norm)
                    X_te, y_te = assemble(te, cond, norm)
                    t0 = time.time()
                    vals = []
                    for seed in SEEDS:
                        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
                        pred = fit_predict_gbdt(Xc, yc, X_te, seed)
                        vals.append(float(np.sqrt(np.mean((y_te - pred) ** 2))))
                    state[key] = float(np.mean(vals))
                    json.dump(state, OUT.open("w"), indent=1)
                    print(f"{key:<26} rmse={state[key]:.5f} "
                          f"[{time.time()-t0:.0f}s]", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    print(f"\n{'='*70}\n2x2 disambiguation: lift delta% "
          f"(negative = better; median over 6 batches [batches improved])")
    print(f"{'cell':<20}{'median delta%':>15}{'batches better':>16}")
    med = {}
    for feat in ("16d", "67d"):
        for norm in ("percell", "source"):
            ds = []
            for b in range(1, 7):
                r0 = state[f"b{b}|{feat}|{norm}|raw"]
                r1 = state[f"b{b}|{feat}|{norm}|srl"]
                ds.append((r1 - r0) / r0 * 100)
            med[(feat, norm)] = float(np.median(ds))
            nb = sum(1 for d in ds if d < 0)
            print(f"{feat}×{norm:<14}{np.median(ds):>+10.1f}{nb:>8}/6")
    d1 = med[("67d", "source")] <= -3 or \
        sum(1 for b in range(1, 7)
            if state[f"b{b}|67d|source|srl"] < state[f"b{b}|67d|source|raw"]) >= 4
    d2 = abs(med[("67d", "percell")]) < 3
    print("\nD1 the lift reappears in 67d x source: "
          + ("holds -- the sparse-features-only explanation is ruled out" if d1
             else "falsified -- narrow the stated scope"))
    print("D2 the marginal result is replicated in 67d x percell: "
          + ("holds" if d2 else "NOT replicated (needs explanation)"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
