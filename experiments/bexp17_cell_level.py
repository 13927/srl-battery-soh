"""bexp17: the cell-level data foundation for conflict gating.

One TabPFN inference is reused for every post-processing variant, and the paired
records are kept per cell:
  (v_pred[Tier-T], corr_mag, v_true[Tier-O], delta% of each constraint)

================ frozen criteria (committed before this run) ================
C1: pooled Spearman rho > 0 with p < 0.01 between cell-level v_pred and delta_iso
C2: the direction is positive within >= 7 of 11 units (within-unit Spearman rho>0)
C3: the predictive power of Tier-T is >= 50 per cent of Tier-O (oracle), as a
    ratio of pooled |Spearman|
If C1 fails, cell-level gating does not hold and the method falls back to
unit-level Tier-S (already verified in bexp13)
=====================================================================

Protocol: reference protocol, history view, context cap 2048 (as bexp12).
Scale: 11 units x 3 seeds, giving records for 100+ test cells x 3 seeds.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.conflict_gating import per_cell_record
from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, assemble_matrix, subsample_context

OUT = Path("results/battery/bexp17_cell_level.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]


def run_unit(unit, seed):
    from residual_lab.models import make_model

    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "author_minmax")
    X_tr, y_tr, _ = assemble_matrix(tr, Xn, "history", "author_minmax")
    X_te, y_te, te_ids = assemble_matrix(te, Xn, "history", "author_minmax")
    X_ctx, y_ctx = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)

    m = make_model("tabpfn_reg", seed=seed)
    m.fit(X_ctx, y_ctx)
    pred = m.predict(X_te)
    rows = per_cell_record(pred, y_te, te_ids)
    for r in rows:
        r.update({"unit": unit.unit_id, "seed": seed, "view": "history"})
    return rows


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["seed"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=[u for u, _, _ in FIT_UNITS])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in args.units:
        ds, bk = umap[uid]
        unit = load_fit_unit(ds, bk)
        for seed in args.seeds:
            if (uid, seed) in done:
                continue
            t0 = time.time()
            rows = run_unit(unit, seed)
            with OUT.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            di = np.mean([r["delta_iso"] for r in rows])
            print(f"{uid:<15}seed={seed} cells={len(rows)} "
                  f"mean_delta_iso={di:+.1f}% [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
