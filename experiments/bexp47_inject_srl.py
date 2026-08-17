"""bexp47: injecting the self-referenced lift into the SOHbenchmark deep baselines
(the injection experiment).

Claim under test: "our representation improves their method" -- add the from_first
offset columns to their handcrafted feature input and see whether their own deep
model (architecture and hyper-parameters unchanged) improves.

Their pipeline (verified against the code, left untouched):
  - features: data/XJTU/handcraft_features/batch-{b}_features.xlsx, one sheet per
    cell, 67 feature columns + a label in the last column
    (XJTU_loader._parser_xlsx takes iloc[:,:-1])
  - normalisation: per-cell min-max(-1,1) (utils/Scaler, fitted per sheet) -- the
    convention under which the lift helped in the 2x2 (per-cell, -6.7 per cent)
  - training: Adam lr=2e-3, MultiStepLR[30,70], n_epoch=100, early_stop=30,
    batch_size=128, validation = a random 10 per cent of the training rows
    (inside _encapsulation)
  - metric: batch-level MSE x1000 = mean_over_test_cells(MSE) x 1000 (the same
    convention as their Appendix C.19 and our Table 6)

Injection (at the data layer, no model code changed):
  for each sheet, append 67 columns of from_first = x_t - x_1 on the raw,
  unnormalised values, keeping the label in the last column; write to a parallel
  data root at results/battery/bexp47_features/. Their Scaler then normalises all
  134 columns per cell as usual -- the pipeline is byte-for-byte unchanged, only
  the input differs.

Two minimal adaptations (both monkeypatched in this script, their files
untouched):
  1) XJTU_loader.root is hard-coded to 'data/XJTU' -> after instantiation the srl
     arm is pointed at the parallel data root
  2) SOHMode._preprocessing_net hard-codes nn.Linear(67, 512) -> parametrised by
     args.feat_dim (equivalent to changing one number in their code)
  Also: they never set a random seed; here both arms are paired on the same seed
  (2023), so the difference comes from the input rather than initialisation luck.
  The device is mps (no CUDA on this machine).

================ frozen criteria (committed before the run) ================
Model: MLP (the fastest of their five baselines). Arms: raw (67d) vs srl (134d).
Scope: all 6 XJTU batches x every test cell per batch. Seeds: 1 (paired,
seed=2023).

D1 (pass): the srl batch-level MSE is below the same-pipeline raw rerun in >= 4 of
    6 batches, and the median improvement over the 6 batches is >= 5 per cent
D2 (refuted): srl is worse in >= 3 of 6 batches -> injection ineffective, no
    extension
D3 (grey zone): in between -> only add seeds to 3 and rerun this stage, no
    extension

Prior, recorded as measured: a smoke test in the other exploration showed that a
trainable network on units with few cells is made worse by the lift (sklearn MLP
on XJTU-2C, +144 per cent); but that was 16 features with no early stopping in
effect, whereas here it is 67 features with their own training recipe, so the
outcome is not predetermined. Either direction is writable: pass -> the
representation improves things regardless of architecture; refuted -> the lift's
benefit depends on the model family, which corroborates the tuned-MLP boundary in
the Limitations.
==============================================================================
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "external" / "SOHbenchmark"
sys.path.insert(0, str(BENCH))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT = ROOT / "results/battery/bexp47_inject.json"
FEAT_ROOT = ROOT / "results/battery/bexp47_features/XJTU"
SEED = 2023
BATCHES = range(1, 7)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------ feature augmentation --
def build_features():
    """Generate the parallel feature files for the srl arm and verify the
    injection."""
    dst = FEAT_ROOT / "handcraft_features"
    dst.mkdir(parents=True, exist_ok=True)
    for b in BATCHES:
        src = BENCH / f"data/XJTU/handcraft_features/batch-{b}_features.xlsx"
        sheets = pd.read_excel(src, sheet_name=None)
        out = {}
        for name, df in sheets.items():
            x = df.iloc[:, :-1]
            ff = x - x.iloc[0]                       # from_first, each cell's own first cycle
            ff.columns = [f"ff_{c}" for c in x.columns]
            out[name] = pd.concat([x, ff, df.iloc[:, -1:]], axis=1)
            # verify: the first offset row is all zeros / label still last / row count unchanged
            assert np.allclose(out[name].iloc[0, 67:134], 0)
            assert out[name].columns[-1] == "label"
            assert len(out[name]) == len(df)
        with pd.ExcelWriter(dst / f"batch-{b}_features.xlsx") as w:
            for name, df in out.items():
                df.to_excel(w, sheet_name=name, index=False)
        print(f"batch-{b}: {len(out)} cells, 67 -> 134 columns", flush=True)


# ------------------------------------------------------------------ training driver --
def make_args(feat_dim):
    import argparse

    a = argparse.Namespace(
        random_seed=SEED, data="XJTU", input_type="handcraft_features",
        batch_size=128, normalized_type="minmax", minmax_range=(-1, 1),
        model="MLP", lr=2e-3, weight_decay=5e-4, n_epoch=100, early_stop=30,
        device=DEVICE, batch=1, feat_dim=feat_dim,
    )
    return a


def patch_pre_net():
    """Parametrise the hard-coded 67 in SOHMode._preprocessing_net (equivalent to
    changing one number in their code)."""
    import nets.Model as M

    def _pre(self):
        if self.args.input_type in ("charge", "partial_charge"):
            return nn.Conv1d(4, 4, kernel_size=1)
        return nn.Linear(getattr(self.args, "feat_dim", 67), 128 * 4)

    M.SOHMode._preprocessing_net = _pre


def run_one(arm, batch, test_id):
    """One full training run for an (arm, batch, test cell); returns the test MSE."""
    from dataloader.XJTU_loader import XJTUDdataset
    from nets.Model import SOHMode

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args = make_args(134 if arm == "srl" else 67)
    args.batch = batch
    loader = XJTUDdataset(args)
    if arm == "srl":
        loader.root = str(FEAT_ROOT)                 # parallel data root, rest unchanged
    dl = loader.get_features(test_battery_id=test_id)
    model = SOHMode(args)
    model.Train(dl["train"], dl["valid"], dl["test"], save_folder=None)
    y, p = model.true_label, model.pred_label
    return float(np.mean((y - p) ** 2))


def main():
    if not (FEAT_ROOT / "handcraft_features" / "batch-1_features.xlsx").exists():
        build_features()
    state = json.load(OUT.open()) if OUT.exists() else {}
    os.chdir(BENCH)                                  # their relative-path convention
    patch_pre_net()
    n_batteries = {1: 8, 2: 15, 3: 8, 4: 8, 5: 8, 6: 8}
    for b in BATCHES:
        for tid in range(1, n_batteries[b] + 1):
            for arm in ("raw", "srl"):
                key = f"b{b}|t{tid}|{arm}"
                if key in state:
                    continue
                t0 = time.time()
                state[key] = run_one(arm, b, tid)
                json.dump(state, OUT.open("w"), indent=1)
                print(f"\n{key:<14} mse={state[key]:.6f} [{time.time()-t0:.0f}s]",
                      flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    n_batteries = {1: 8, 2: 15, 3: 8, 4: 8, 5: 8, 6: 8}
    print(f"\n{'='*74}\ninjection verdict (their MLP, default hyper-parameters, "
          f"paired seed {SEED})  metric: batch-level MSE x1000")
    print(f"{'batch':<8}{'raw':>10}{'srl':>10}{'delta%':>9}{'cells':>10}")
    deltas, better = [], 0
    for b in BATCHES:
        raws = [state[f"b{b}|t{t}|raw"] for t in range(1, n_batteries[b] + 1)
                if f"b{b}|t{t}|raw" in state]
        srls = [state[f"b{b}|t{t}|srl"] for t in range(1, n_batteries[b] + 1)
                if f"b{b}|t{t}|srl" in state]
        if not raws or len(raws) != len(srls):
            print(f"b{b}: incomplete ({len(raws)}/{n_batteries[b]})")
            continue
        r, s = np.mean(raws) * 1000, np.mean(srls) * 1000
        d = (s - r) / r * 100
        deltas.append(d)
        better += int(s < r)
        print(f"b{b:<7}{r:>10.4f}{s:>10.4f}{d:>+9.1f}{len(raws):>8}/{n_batteries[b]}")
    if len(deltas) < 6:
        print("(not all 6 batches finished; verdict deferred)")
        return
    med = float(np.median(deltas))
    worse = sum(d > 0 for d in deltas)
    d1 = better >= 4 and med <= -5
    d2 = worse >= 3
    print(f"\nbatches where srl is better = {better}/6;  median delta% = {med:+.1f}%")
    print("D1 pass (>=4/6 and median <=-5%): "
          + ("holds" if d1 else "no"))
    print("D2 refuted (>=3/6 worse): "
          + ("holds" if d2 else "no"))
    if d1:
        v = "pass -> the representation improves things regardless of architecture"
    elif d2:
        v = "refuted -> the benefit depends on the model family, matching the MLP boundary"
    else:
        v = "grey zone -> add seeds to 3 and rerun per D3, no extension"
    print(f"verdict: {v}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--features" in sys.argv:
        build_features()
    elif "--smoke" in sys.argv:
        build_features()
        os.chdir(BENCH)
        patch_pre_net()
        for arm in ("raw", "srl"):
            t0 = time.time()
            mse = run_one(arm, 1, 1)
            print(f"\nsmoke b1|t1|{arm}: mse={mse:.6f} [{time.time()-t0:.0f}s]",
                  flush=True)
    else:
        main()
