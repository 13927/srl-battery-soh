"""bexp27: randomisation inference over many derangements (a Fisher permutation
test).

bexp26 established the direction with a single derangement; here 20 different
derangements build a permutation null distribution, turning "the self-reference is
causal" into a formal permutation p value.

Statistic: T_m = mean_units(RMSE_shuffled_m) for each derangement m, compared with
T_self.
H0: the identity of the reference is irrelevant, so self should land in the middle
of the permutation distribution.

================ frozen criteria (committed before the run) ================
R1 (pooled): permutation p = (1 + #{m: T_m <= T_self}) / (M+1) < 0.05
           with M=20 this requires that no derangement beats self on the mean over
           units (p = 1/21 ~ 0.048)
R2 (per unit): p_unit = (1 + #{m: rmse_m,u <= rmse_self,u}) / 21 <= 0.1
           (that is, at most one derangement beats self in that unit) in >= 8 of 11
           units
=======================================================

Derangements: a random permutation from each of the seeds 1..20 followed by a
cyclic shift (never any fixed point; distinctness is checked one by one).
GBDT, reference protocol, cap=2048, 5 seeds. The self and raw arms are recomputed
here as well, so the script is self-contained and cross-checks bexp26.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, fit_predict_gbdt, subsample_context
from battery_lab.temporal_features import cycle_author_minmax

OUT = Path("results/battery/bexp27_randomization.json")
SEEDS = [0, 1, 2, 3, 4]
MAP_SEEDS = list(range(1, 21))       # 20 derangements


def derangement(cell_ids, map_seed):
    rng = np.random.default_rng(map_seed)
    perm = list(rng.permutation(cell_ids))
    return {perm[i]: perm[(i + 1) % len(perm)] for i in range(len(perm))}


def assemble(cells, Xn, mode, ref_map=None):
    xs, ys = [], []
    for c in cells:
        Z = Xn[c.cell_id]
        cyc = cycle_author_minmax(len(Z))
        base = [Z, cyc]
        if mode == "self":
            base.append(Z - Z[:1])
        elif mode == "shuffled":
            base.append(Z - Xn[ref_map[c.cell_id]][:1])
        xs.append(np.hstack(base))
        ys.append(c.y)
    return np.vstack(xs), np.concatenate(ys)


def eval_mode(unit, Xn, mode, ref_map=None):
    X_tr, y_tr = assemble(unit.train_cells, Xn, mode, ref_map)
    X_te, y_te = assemble(unit.test_cells, Xn, mode, ref_map)
    vals = []
    for seed in SEEDS:
        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
        pred = fit_predict_gbdt(Xc, yc, X_te, seed)
        vals.append(float(np.sqrt(np.mean((y_te - pred) ** 2))))
    return float(np.mean(vals))


def main():
    state = json.load(OUT.open()) if OUT.exists() else {}
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in [u for u, _, _ in FIT_UNITS]:
        st = state.setdefault(uid, {})
        unit = None
        todo = (["self"] if "self" not in st else []) + \
               [f"perm{m}" for m in MAP_SEEDS if f"perm{m}" not in st]
        if not todo:
            continue
        unit = load_fit_unit(*umap[uid])
        cells = unit.train_cells + unit.test_cells
        Xn = normalize_cells(cells, "author_minmax")
        ids = [c.cell_id for c in cells]
        # check that the derangements are distinct
        maps = {m: derangement(ids, m) for m in MAP_SEEDS}
        sigs = {m: tuple(sorted(maps[m].items())) for m in MAP_SEEDS}
        assert len(set(sigs.values())) == len(MAP_SEEDS), "duplicate derangement"
        assert all(k != v for m in MAP_SEEDS for k, v in maps[m].items()), "fixed point found"
        t0 = time.time()
        for task in todo:
            if task == "self":
                st["self"] = eval_mode(unit, Xn, "self")
            else:
                m = int(task[4:])
                st[task] = eval_mode(unit, Xn, "shuffled", maps[m])
            json.dump(state, OUT.open("w"), indent=1)
        print(f"{uid:<15} self={st['self']:.5f} "
              f"perms={len([k for k in st if k.startswith('perm')])}/20 "
              f"[{time.time()-t0:.0f}s]", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    units = [u for u, _, _ in FIT_UNITS if u in state and "self" in state[u]]
    M = len(MAP_SEEDS)
    print(f"\n{'='*76}\npermutation inference "
          f"({M} derangements, GBDT, mean of 5 seeds)")
    print(f"{'unit':<16}{'self':>9}{'perm med':>10}{'perm best':>11}"
          f"{'beat self':>11}{'p_unit':>8}")
    n_r2 = 0
    T_perm = np.zeros(M)
    T_self = 0.0
    for uid in units:
        st = state[uid]
        perms = np.array([st[f"perm{m}"] for m in MAP_SEEDS])
        beat = int((perms <= st["self"]).sum())
        p_u = (1 + beat) / (M + 1)
        if p_u <= 0.1:
            n_r2 += 1
        T_perm += perms / len(units)
        T_self += st["self"] / len(units)
        print(f"{uid:<16}{st['self']:>9.5f}{np.median(perms):>10.5f}"
              f"{perms.min():>10.5f}{beat:>12}{p_u:>8.3f}")
    beat_pool = int((T_perm <= T_self).sum())
    p_pool = (1 + beat_pool) / (M + 1)
    print(f"\npooled: T_self={T_self:.5f}  "
          f"T_perm median={np.median(T_perm):.5f} "
          f"range [{T_perm.min():.5f}, {T_perm.max():.5f}]")
    print(f"R1 permutation p = {p_pool:.3f} (criterion <0.05) -> "
          + ("holds" if p_pool < 0.05 else "falsified"))
    print(f"R2 units with p_unit<=0.1 = {n_r2}/{len(units)} (criterion >=8) -> "
          + ("holds" if n_r2 >= 8 else "falsified"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
