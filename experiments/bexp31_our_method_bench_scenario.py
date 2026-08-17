"""bexp31: our method inside the official SOHbenchmark scenario (their published
numbers are cited, their baselines are not re-run).

Scenario (reproducing the semantics of their XJTU_loader.get_features exactly):
- split: leave-one-cell-out within a batch (every cell of a batch is the test cell
  in turn)
- features: their 67 handcraft_features, min-max scaled to (-1,1) per cell (their
  _parser_xlsx)
- label: label / 2.0 (their max_capacity)
- opponent numbers: Appendix Table C.19, group C (MSE x1000, mean of three runs,
  averaged over the cells of a batch)

Our two conditions (the only variable is the self-referenced lift):
  raw      67d = the very matrix they feed to the five deep models
  srl      134d = raw + (X_t - X_first_own), the from_first lift, legal in
                  deployment

Models: GBDT (reference-protocol hyper-parameters) and TabPFN (training-free,
context cap 2048), three seeds each.
Metric: whole-life MSE per cell, averaged over the cells of a batch, x1000 (the
same construction as C.19; RMSE is also reported).
A descriptive comparison with no selection step; the question is whether the lift
carries an off-the-shelf model into or past the best of their five baselines.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from battery_lab.protocols import (CONTEXT_CAP, fit_predict_gbdt,
                                   fit_predict_tabpfn, subsample_context)

BENCH = ROOT / "external/SOHbenchmark/data/XJTU/handcraft_features"
OUT = ROOT / "results/battery/bexp31_our_method_bench_scenario.json"
SEEDS = [0, 1, 2]
BATCHES = [1, 2, 3, 4, 5, 6]
MAX_CAP = 2.0

# Table C.19, input type C (handcraft, [-1,1]): batch-level MSE x1000, models in
# the order CNN/LSTM/GRU/MLP/Attention
C19_MSE = {
    1: [0.060, 0.039, 0.028, 0.034, 0.078],
    2: [0.294, 0.135, 0.109, 0.104, 0.171],
    3: [0.042, 0.039, 0.026, 0.024, 0.058],
    4: [0.045, 0.057, 0.045, 0.038, 0.065],
    5: [0.237, 0.172, 0.154, 0.451, 0.180],
    6: [0.171, 0.248, 0.218, 0.214, 0.227],
}
MODELS = ["CNN", "LSTM", "GRU", "MLP", "Attention"]


def load_batch(batch):
    """Return [(X_norm_67d, soh), ...] per cell, reproducing _parser_xlsx."""
    dfs = pd.read_excel(BENCH / f"batch-{batch}_features.xlsx", sheet_name=None)
    cells = []
    for name, df in dfs.items():
        x = np.array(df.iloc[:, :-1], dtype=np.float64)
        y = np.array(df["label"], dtype=np.float64) / MAX_CAP
        mx, mn = x.max(axis=0, keepdims=True), x.min(axis=0, keepdims=True)
        span = np.where(mx - mn == 0, 1.0, mx - mn)
        xn = (x - mn) / span * 2.0 - 1.0          # per-cell min-max to (-1,1)
        cells.append((xn, y))
    return cells


def features(xn, cond):
    if cond == "raw":
        return xn
    return np.hstack([xn, xn - xn[:1]])            # srl: + from_first(own)


def run():
    state = json.load(OUT.open()) if OUT.exists() else {}
    for batch in BATCHES:
        cells = load_batch(batch)
        for ti in range(len(cells)):
            for cond in ("raw", "srl"):
                for backbone, fit in (("gbdt", fit_predict_gbdt),
                                      ("tabpfn", fit_predict_tabpfn)):
                    key = f"b{batch}|t{ti+1}|{cond}|{backbone}"
                    if key in state:
                        continue
                    X_te = features(cells[ti][0], cond)
                    y_te = cells[ti][1]
                    X_tr = np.vstack([features(c[0], cond)
                                      for j, c in enumerate(cells) if j != ti])
                    y_tr = np.concatenate([c[1] for j, c in enumerate(cells)
                                           if j != ti])
                    t0 = time.time()
                    mses = []
                    for seed in SEEDS:
                        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
                        pred = fit(Xc, yc, X_te, seed)
                        mses.append(float(np.mean((y_te - pred) ** 2)))
                    state[key] = {"mse": float(np.mean(mses)), "seed_mse": mses}
                    json.dump(state, OUT.open("w"), indent=1)
                    print(f"{key:<24} MSEx1000={state[key]['mse']*1000:.3f} "
                          f"[{time.time()-t0:.0f}s]", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    n_cells = {b: len(load_batch(b)) for b in BATCHES}
    print(f"\n{'='*100}\nbatch-level MSE x1000 "
          f"(same construction as C.19; RMSE in brackets)")
    hdr = f"{'batch':<6}" + "".join(f"{m:>9}" for m in MODELS) + \
          f"{'|':>3}{'G-raw':>9}{'G-srl':>9}{'T-raw':>9}{'T-srl':>9}"
    print(hdr)
    win = {k: 0 for k in ("G-srl", "T-srl")}
    for b in BATCHES:
        best_bench = min(C19_MSE[b])
        row = f"{b:<4}" + "".join(f"{v:>9.3f}" for v in C19_MSE[b]) + f"{'|':>3}"
        for tag, cond, bk in (("G-raw", "raw", "gbdt"), ("G-srl", "srl", "gbdt"),
                              ("T-raw", "raw", "tabpfn"), ("T-srl", "srl", "tabpfn")):
            ks = [f"b{b}|t{t+1}|{cond}|{bk}" for t in range(n_cells[b])]
            if all(k in state for k in ks):
                v = np.mean([state[k]["mse"] for k in ks]) * 1000
                row += f"{v:>9.3f}"
                if tag in win and v < best_bench:
                    win[tag] += 1
            else:
                row += f"{'--':>9}"
        print(row)
    print(f"\nbatches where the lift beats the best of their five baselines: "
          f"GBDT {win['G-srl']}/6, TabPFN {win['T-srl']}/6")
    print("note: their numbers are published means of three runs, cited directly; "
          "ours are means of three seeds; the 2048 context cap is self-imposed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        run()
