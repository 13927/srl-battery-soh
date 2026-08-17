"""bexp32: refine the permutation test to 100 derangements.

Reuses the bexp27 infrastructure: the self arm and derangements 1..20 are taken
from its result file, and this script only adds the shuffled arm for derangement
seeds 21..100 (GBDT, 11 units, 5 seeds).

================ frozen criteria (committed before the run) ================
R3: with all 100 derangements pooled, the permutation p value
    p = (1 + #{m: T_m <= T_self}) / 101 < 0.05
    (if the self arm still wins everywhere, p = 1/101 ~ 0.0099); the exact p is
    reported as measured
=======================================================
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from bexp27_randomization import OUT, derangement, eval_mode

NEW_SEEDS = list(range(21, 101))


def main():
    state = json.load(OUT.open())
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in [u for u, _, _ in FIT_UNITS]:
        st = state[uid]
        todo = [m for m in NEW_SEEDS if f"perm{m}" not in st]
        if not todo:
            continue
        unit = load_fit_unit(*umap[uid])
        cells = unit.train_cells + unit.test_cells
        Xn = normalize_cells(cells, "author_minmax")
        ids = [c.cell_id for c in cells]
        t0 = time.time()
        for m in todo:
            st[f"perm{m}"] = eval_mode(unit, Xn, "shuffled", derangement(ids, m))
            json.dump(state, OUT.open("w"), indent=1)
        print(f"{uid:<15} perms={sum(1 for k in st if k.startswith('perm'))}/100 "
              f"[{time.time()-t0:.0f}s]", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    M = 100
    units = [u for u, _, _ in FIT_UNITS]
    T_perm = np.zeros(M)
    T_self = 0.0
    n_unit_top = 0
    for uid in units:
        st = state[uid]
        perms = np.array([st[f"perm{m}"] for m in range(1, M + 1)])
        T_perm += perms / len(units)
        T_self += st["self"] / len(units)
        beat = int((perms <= st["self"]).sum())
        if (1 + beat) / (M + 1) <= 0.1:
            n_unit_top += 1
    beat_pool = int((T_perm <= T_self).sum())
    p = (1 + beat_pool) / (M + 1)
    print(f"\n100 derangements: T_self={T_self:.5f} "
          f"null median={np.median(T_perm):.5f} "
          f"range [{T_perm.min():.5f}, {T_perm.max():.5f}]")
    print(f"R3 exact permutation p = {p:.4f} (criterion <0.05) -> "
          f"{'holds' if p < 0.05 else 'falsified'}")
    print(f"reference: units with p_unit<=0.1 = {n_unit_top}/11")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
