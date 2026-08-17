"""bexp43: permutation inference on the TabPFN side, and bexp44: the three arms at
20 seeds.

--- bexp43 (--tabpfn) ---
This completes the claim that both model families carry a causal p value: the GBDT
side already had 100 derangements with p=0.0099 (bexp27/32), while the TabPFN side
had only three units and a single derangement (bexp26). Here the formal test is
run.

Representative units: TJU-1 (from_first carries 46 per cent of the importance),
MIT (89 per cent), HUST (the cycle column carries 99 per cent) -- covering both
carrier regimes.
Arms: self plus 20 derangements (seeds 1..20, reusing bexp27.derangement, the same
permutations as on the GBDT side).
TabPFN inference is frozen, cap=2048, 3 seeds, reference protocol (author_minmax +
cycle_author_minmax).

================ frozen criteria (committed before the run) ================
P1a (main): permutation of the pooled statistic T = mean_units(RMSE) over the
three units
          p = (1 + #{m: T_m <= T_self}) / 21 < 0.05
P1b (per unit): the self arm beats >= 18 of 20 derangements in >= 2 of 3 units
If it fails: record the outcome as measured and narrow the causal wording to
          "a formal p value for GBDT, directional evidence for TabPFN"; the claim
          that both families carry a causal p value may not be made.
=======================================================

--- bexp44 (--seeds20) ---
The three arms of bexp26 on all 11 units, with the GBDT seeds raised from 5 to 20.
Criterion B1: at 20 seeds, self < shuffled in >= 9 of 11 units (replicating the
5-seed result).
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import (CONTEXT_CAP, fit_predict_gbdt,
                                   fit_predict_tabpfn, subsample_context)
from battery_lab.temporal_features import cycle_author_minmax
from bexp27_randomization import derangement

OUT_T = ROOT / "results/battery/bexp43_tabpfn_perm.json"
OUT_S = ROOT / "results/battery/bexp44_seeds20.json"
TAB_UNITS = ["TJU-1", "MIT", "HUST"]
TAB_SEEDS = [0, 1, 2]
MAP_SEEDS = list(range(1, 21))
SEEDS20 = list(range(20))


def assemble(cells, Xn, arm, ref_map=None):
    xs, ys = [], []
    for c in cells:
        Z = Xn[c.cell_id]
        cols = [Z, cycle_author_minmax(len(Z))]
        if arm == "self":
            cols.append(Z - Z[:1])
        else:
            cols.append(Z - Xn[ref_map[c.cell_id]][:1])
        xs.append(np.hstack(cols))
        ys.append(c.y)
    return np.vstack(xs), np.concatenate(ys)


def eval_arm(unit, Xn, arm, fit, seeds, ref_map=None):
    X_tr, y_tr = assemble(unit.train_cells, Xn, arm, ref_map)
    X_te, y_te = assemble(unit.test_cells, Xn, arm, ref_map)
    vals = []
    for s in seeds:
        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, s)
        vals.append(float(np.sqrt(np.mean((y_te - fit(Xc, yc, X_te, s)) ** 2))))
    return float(np.mean(vals))


def run_tabpfn():
    state = json.load(OUT_T.open()) if OUT_T.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in TAB_UNITS:
        unit = load_fit_unit(*umap[uid])
        Xn = normalize_cells(unit.cells, "author_minmax")
        ids = [c.cell_id for c in unit.cells]
        maps = {m: derangement(ids, m) for m in MAP_SEEDS}
        todo = ([("self", None)] if f"{uid}|self" not in state else []) + \
               [(f"perm{m}", maps[m]) for m in MAP_SEEDS
                if f"{uid}|perm{m}" not in state]
        for name, rm in todo:
            t0 = time.time()
            arm = "self" if name == "self" else "shuffled"
            state[f"{uid}|{name}"] = eval_arm(unit, Xn, arm, fit_predict_tabpfn,
                                              TAB_SEEDS, rm)
            json.dump(state, OUT_T.open("w"), indent=1)
            print(f"{uid:<8} {name:<8} rmse={state[f'{uid}|{name}']:.5f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    report_tabpfn(state)


def report_tabpfn(state=None):
    state = state or json.load(OUT_T.open())
    M = len(MAP_SEEDS)
    print(f"\n{'='*70}\nTabPFN permutation test "
          f"({M} derangements, 3 units, 3 seeds)")
    T_perm = np.zeros(M); T_self = 0.0; n_unit = 0; n_pass_unit = 0
    for uid in TAB_UNITS:
        if f"{uid}|self" not in state:
            continue
        perms = np.array([state.get(f"{uid}|perm{m}", np.nan) for m in MAP_SEEDS])
        if np.isnan(perms).any():
            print(f"{uid}: incomplete ({np.sum(~np.isnan(perms))}/{M})")
            continue
        n_unit += 1
        beat = int((perms > state[f"{uid}|self"]).sum())      # derangements beaten
        if beat >= 18:
            n_pass_unit += 1
        T_perm += perms; T_self += state[f"{uid}|self"]
        print(f"{uid:<8} self={state[f'{uid}|self']:.5f} "
              f"perm median={np.median(perms):.5f} self beats {beat}/{M}")
    if n_unit:
        T_perm /= n_unit; T_self /= n_unit
        p = (1 + int((T_perm <= T_self).sum())) / (M + 1)
        print(f"\npooled: T_self={T_self:.5f} "
              f"null median={np.median(T_perm):.5f} "
              f"range [{T_perm.min():.5f}, {T_perm.max():.5f}]")
        print(f"P1a permutation p={p:.4f} (criterion <0.05) -> "
              + ("holds" if p < 0.05 else "falsified"))
        print(f"P1b units where self beats >=18/20: {n_pass_unit}/{n_unit} "
              f"(criterion >=2) -> "
              + ("holds" if n_pass_unit >= 2 else "falsified"))


def run_seeds20():
    state = json.load(OUT_S.open()) if OUT_S.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in [u for u, _, _ in FIT_UNITS]:
        if f"{uid}|shuffled" in state:
            continue
        unit = load_fit_unit(*umap[uid])
        Xn = normalize_cells(unit.cells, "author_minmax")
        rm = derangement([c.cell_id for c in unit.cells], 20260731)
        t0 = time.time()
        for arm in ("self", "shuffled"):
            state[f"{uid}|{arm}"] = eval_arm(unit, Xn, arm, fit_predict_gbdt,
                                             SEEDS20, rm if arm == "shuffled" else None)
        json.dump(state, OUT_S.open("w"), indent=1)
        print(f"{uid:<16} self={state[f'{uid}|self']:.5f} "
              f"shuf={state[f'{uid}|shuffled']:.5f} [{time.time()-t0:.0f}s]", flush=True)
    report_seeds20(state)


def report_seeds20(state=None):
    state = state or json.load(OUT_S.open())
    n = 0; tot = 0
    print(f"\nthree arms at 20 seeds (GBDT)")
    for uid, _, _ in FIT_UNITS:
        if f"{uid}|self" not in state:
            continue
        tot += 1
        ok = state[f"{uid}|self"] < state[f"{uid}|shuffled"]
        n += int(ok)
        print(f"  {uid:<16} self={state[f'{uid}|self']:.5f} "
              f"shuf={state[f'{uid}|shuffled']:.5f} {'✓' if ok else '✗'}")
    print(f"B1 self<shuffled: {n}/{tot} (criterion >=9) -> "
          + ("holds" if n >= 9 else "falsified"))


if __name__ == "__main__":
    if "--tabpfn" in sys.argv:
        run_tabpfn()
    elif "--seeds20" in sys.argv:
        run_seeds20()
    else:
        report_tabpfn(); report_seeds20()
