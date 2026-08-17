"""bexp26: the decisive experiment -- self-reference against mispaired reference.

Background: the ablation established that from_first (X_t - X_first_own) is the
dominant family. But "it works" admits two readings:
  H_self:  what matters is the self-pairing -- subtracting this cell's own
           beginning-of-life state (the self-reference account holds)
  H_trend: it is only drift or trend information under an arbitrary reference, and
           "own" is irrelevant (the account fails)

Control design (GBDT, reference protocol, cap=2048, 5 seeds, 11 units):
  raw            17d baseline
  self           raw + (X_t - X_first_own)        33d, the incumbent from_first
  shuffled       raw + (X_t - X_first_other)      33d, the reference replaced by a
                 fixed, mispaired other cell -- identical width and marginal
                 statistics, only the self-pairing is broken
  pop            raw + (X_t - mean_src(X_first))  33d, a population-mean reference
                 -- an affine shift for tree models, so it should match raw (a
                 sanity control on the apparatus)

================ frozen criteria (committed before the run) ================
J1 (main): with GBDT, self beats shuffled in >= 8 of 11 units
           -> holds: the self-pairing is causal and the account becomes a finding
           -> fails: the account fails, record it as measured, and describe
              from_first as a drift feature only
J2 (apparatus): the median |delta%| of pop against raw is < 2 per cent (affine
           invariance sanity check)
J3 (across models): spot-check TabPFN on TJU-1/HUST/MIT x 3 seeds; self beats
           shuffled in >= 2 of 3
=======================================================

The mispairing: a cyclic-shift permutation with the fixed seed 20260731 (no fixed
points by construction), recorded in the result file.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import (CONTEXT_CAP, fit_predict_gbdt,
                                   fit_predict_tabpfn, subsample_context)
from battery_lab.temporal_features import cycle_author_minmax

OUT = Path("results/battery/bexp26_self_vs_population.json")
SEEDS = [0, 1, 2, 3, 4]
TAB_UNITS = ["TJU-1", "HUST", "MIT"]
TAB_SEEDS = [0, 1, 2]
MAP_SEED = 20260731


def derangement(cell_ids):
    """Cyclic-shift derangement from a fixed seed (no fixed points)."""
    rng = np.random.default_rng(MAP_SEED)
    perm = list(rng.permutation(cell_ids))
    return {perm[i]: perm[(i + 1) % len(perm)] for i in range(len(perm))}


def assemble(cells, Xn, mode, ref_map=None, pop_first=None):
    xs, ys = [], []
    for c in cells:
        Z = Xn[c.cell_id]
        cyc = cycle_author_minmax(len(Z))
        base = [Z, cyc]
        if mode == "self":
            base.append(Z - Z[:1])
        elif mode == "shuffled":
            ref = Xn[ref_map[c.cell_id]][:1]
            base.append(Z - ref)
        elif mode == "pop":
            base.append(Z - pop_first)
        xs.append(np.hstack(base))
        ys.append(c.y)
    return np.vstack(xs), np.concatenate(ys)


def run_unit(unit, backbone, seeds):
    cells = unit.train_cells + unit.test_cells
    Xn = normalize_cells(cells, "author_minmax")
    ref_map = derangement([c.cell_id for c in cells])
    pop_first = np.mean([Xn[c.cell_id][0] for c in unit.train_cells], axis=0,
                        keepdims=True)                     # source cells only (legal)
    fit = fit_predict_gbdt if backbone == "gbdt" else fit_predict_tabpfn
    out = {}
    modes = ("raw", "self", "shuffled", "pop") if backbone == "gbdt" \
        else ("raw", "self", "shuffled")
    for mode in modes:
        X_tr, y_tr = assemble(unit.train_cells, Xn, mode, ref_map, pop_first)
        X_te, y_te = assemble(unit.test_cells, Xn, mode, ref_map, pop_first)
        vals = []
        for seed in seeds:
            Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
            pred = fit(Xc, yc, X_te, seed)
            vals.append(float(np.sqrt(np.mean((y_te - pred) ** 2))))
        out[mode] = {"rmse": float(np.mean(vals)), "seed_vals": vals}
    out["ref_map"] = ref_map
    return out


def main():
    state = json.load(OUT.open()) if OUT.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}

    for uid in [u for u, _, _ in FIT_UNITS]:
        if f"gbdt::{uid}" in state:
            continue
        t0 = time.time()
        r = run_unit(load_fit_unit(*umap[uid]), "gbdt", SEEDS)
        state[f"gbdt::{uid}"] = r
        json.dump(state, OUT.open("w"), indent=1)
        print(f"gbdt   {uid:<15} raw={r['raw']['rmse']:.5f} "
              f"self={r['self']['rmse']:.5f} shuf={r['shuffled']['rmse']:.5f} "
              f"pop={r['pop']['rmse']:.5f} [{time.time()-t0:.0f}s]", flush=True)

    for uid in TAB_UNITS:
        if f"tabpfn::{uid}" in state:
            continue
        t0 = time.time()
        r = run_unit(load_fit_unit(*umap[uid]), "tabpfn", TAB_SEEDS)
        state[f"tabpfn::{uid}"] = r
        json.dump(state, OUT.open("w"), indent=1)
        print(f"tabpfn {uid:<15} raw={r['raw']['rmse']:.5f} "
              f"self={r['self']['rmse']:.5f} shuf={r['shuffled']['rmse']:.5f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    print(f"\n{'='*80}\nverdict (delta% against raw, negative = better)")
    print(f"{'unit':<16}{'self%':>9}{'shuffled%':>10}{'pop%':>8}{'self<shuf?':>11}")
    j1 = 0
    pop_abs = []
    for uid, _, _ in FIT_UNITS:
        r = state.get(f"gbdt::{uid}")
        if not r:
            continue
        b = r["raw"]["rmse"]
        ds = (r["self"]["rmse"] - b) / b * 100
        dh = (r["shuffled"]["rmse"] - b) / b * 100
        dp = (r["pop"]["rmse"] - b) / b * 100
        pop_abs.append(abs(dp))
        win = r["self"]["rmse"] < r["shuffled"]["rmse"]
        j1 += int(win)
        print(f"{uid:<16}{ds:>+9.1f}{dh:>+10.1f}{dp:>+8.1f}{'✓' if win else '':>10}")
    print(f"\nJ1 self<shuffled: {j1}/11 (criterion >=8) -> "
          + ("holds -- the self-pairing is causal" if j1 >= 8
             else "falsified -- the account fails"))
    print(f"J2 median |pop delta%|={np.median(pop_abs):.2f}% (criterion <2) -> "
          + ("apparatus sound" if np.median(pop_abs) < 2
             else "APPARATUS ANOMALY"))
    j3 = 0
    for uid in TAB_UNITS:
        r = state.get(f"tabpfn::{uid}")
        if not r:
            continue
        win = r["self"]["rmse"] < r["shuffled"]["rmse"]
        j3 += int(win)
        print(f"J3 tabpfn {uid:<10} self={r['self']['rmse']:.5f} "
              f"shuf={r['shuffled']['rmse']:.5f} {'✓' if win else '✗'}")
    print(f"J3 TabPFN spot-check: {j3}/{len(TAB_UNITS)} (criterion >=2)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
