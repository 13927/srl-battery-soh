"""bexp30: reproduce the five deep baselines of SOHbenchmark
(CNN/LSTM/GRU/MLP/Attention).

Source: wang-fujin/SOHbenchmark, the official benchmark of the PINN4SOH authors
(J. Energy Storage 2024).
Purpose: widen the comparison from one PINN to five further standard deep models,
all under the authors' own protocol.

Alignment (a descriptive comparison: no selection step, so no frozen criterion is
needed):
- data=XJTU, input_type=handcraft_features (the same 16 statistics as PINN4SOH and
  as this work)
- normalized_type=minmax, range=(-1,1), as in the reference
- test cells = the official PINN4SOH split (file names containing 4/8: [4,8] per
  batch, plus 14 in the 15-cell batch)
- device=cpu; three random seeds (2023/2024/2025; their main defaults to five
  repetitions that rely on CUDA non-determinism, so on CPU the seed has to be
  varied explicitly -- the deviation is recorded as such)
- metric: per-test-cell RMSE averaged within a batch (the authors' cell-macro
  convention)
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "external/SOHbenchmark"
sys.path.insert(0, str(BENCH))

import numpy as np
import torch

OUT = ROOT / "results/battery/bexp30_sohbenchmark.json"
MODELS = ["CNN", "LSTM", "GRU", "MLP", "Attention"]
SEEDS = [2023, 2024, 2025]
# official PINN4SOH test cell ids (file names containing '4' or '8');
# batch 2 has 15 cells, so 14 is included
TEST_IDS = {1: [4, 8], 2: [4, 8, 14], 3: [4, 8], 4: [4, 8], 5: [4, 8], 6: [4, 8]}
BATCH_NAME = {1: "XJTU-2C", 2: "XJTU-3C", 3: "XJTU-R2.5", 4: "XJTU-R3",
              5: "XJTU-RW", 6: "XJTU-satellite"}


def get_args():
    import main as bench_main
    args = bench_main.get_args()
    args.data = "XJTU"
    args.input_type = "handcraft_features"
    args.normalized_type = "minmax"
    args.minmax_range = (-1, 1)
    args.device = "cpu"
    args.save_folder = "/tmp/sohbench_runs"
    return args


def run_one(model_name, batch, test_id, seed):
    import main as bench_main
    from nets.Model import SOHMode

    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    args = get_args()
    args.model = model_name
    args.batch = batch
    args.random_seed = seed
    dl = bench_main.load_data(args, test_battery_id=test_id)
    m = SOHMode(args)
    m.Train(dl["train"], dl["valid"], dl["test"], save_folder=None)
    t, p = np.asarray(m.true_label).ravel(), np.asarray(m.pred_label).ravel()
    return float(np.sqrt(np.mean((t - p) ** 2)))


def main():
    import os
    os.chdir(BENCH)          # their dataloader uses the relative path data/
    state = json.load(OUT.open()) if OUT.exists() else {}
    for model_name in MODELS:
        for batch, ids in TEST_IDS.items():
            for tid in ids:
                for seed in SEEDS:
                    key = f"{model_name}|b{batch}|t{tid}|s{seed}"
                    if key in state:
                        continue
                    t0 = time.time()
                    try:
                        rmse = run_one(model_name, batch, tid, seed)
                        state[key] = rmse
                        print(f"{key:<28} rmse={rmse:.6f} "
                              f"[{time.time()-t0:.0f}s]", flush=True)
                    except Exception as ex:
                        state[key] = f"ERROR: {type(ex).__name__}: {ex}"
                        print(f"{key:<28} {state[key]}", flush=True)
                    json.dump(state, OUT.open("w"), indent=1)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    print(f"\n{'='*80}\nSOHbenchmark baselines "
          f"(cell-macro, mean over 3 seeds)")
    print(f"{'unit':<16}" + "".join(f"{m:>10}" for m in MODELS))
    unit_means = {m: [] for m in MODELS}
    for batch, ids in TEST_IDS.items():
        line = f"{BATCH_NAME[batch]:<16}"
        for m in MODELS:
            vals = [state[f"{m}|b{batch}|t{t}|s{s}"] for t in ids for s in SEEDS
                    if isinstance(state.get(f"{m}|b{batch}|t{t}|s{s}"), float)]
            if vals:
                # average seeds per cell first, then take the macro mean
                per_cell = [np.mean([state[f"{m}|b{batch}|t{t}|s{s}"]
                                     for s in SEEDS
                                     if isinstance(state.get(f"{m}|b{batch}|t{t}|s{s}"), float)])
                            for t in ids]
                v = float(np.mean(per_cell))
                unit_means[m].append(v)
                line += f"{v:>10.5f}"
            else:
                line += f"{'--':>10}"
        print(line)
    print(f"{'XJTU mean':<15}" + "".join(
        f"{np.mean(unit_means[m]):>10.5f}" if unit_means[m] else f"{'--':>10}"
        for m in MODELS))
    print("\nreference: published PINN (all batches) 0.010879 | "
          "TabPFN+history 0.007932 | GBDT+history 0.008146")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
