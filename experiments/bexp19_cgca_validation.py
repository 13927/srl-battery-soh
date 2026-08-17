"""bexp19: frozen verification of the two-signal decomposition on the unexposed
library (MATR-CLO).

Frozen predictions: F1 v_pred ~ delta_iso, rho<0; F2 roughness_ratio ~
          delta_smooth, rho<0; F3 the Tier-S gate (r_src=0.075<0.15) applies iso
          with a unit-level delta in [-14.6, +5.0]; F4 after gating, no worse than
          never beyond the noise floor.
Run once, 3 seeds.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

from battery_lab.conflict_gating import per_cell_record
from bexp5_unexposed_frozen import load_unit_npz, run_view

OUT = Path("results/battery/bexp19_cgca_validation.json")
SEEDS = [0, 1, 2]
R_SRC = 0.0750          # frozen bexp13 value
REF_ROUGH_SRC = None    # source-label roughness, computed from the source side at run time (legal)


def main():
    unit = load_unit_npz()
    src_rough = float(np.median([np.std(np.diff(c.y)) for c in unit.train_cells
                                 if len(c.y) > 2]))
    all_rows = []
    for seed in SEEDS:
        _, pred, y_te, te_ids = run_view(unit, "history", seed)
        rows = per_cell_record(pred, y_te, te_ids)
        for r in rows:
            r["seed"] = seed
            r["rough_ratio"] = r["rough_pred"] / max(src_rough, 1e-9)
        all_rows.extend(rows)

    # aggregate per cell (mean of 3 seeds)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_rows:
        agg[r["cell_id"]].append(r)
    cells = []
    for c, rs in agg.items():
        cells.append({k: float(np.mean([x[k] for x in rs]))
                      for k in ("v_pred", "rough_ratio", "delta_iso",
                                "delta_smooth", "rmse_none", "rmse_iso")}
                     | {"cell": c})

    vp = [c["v_pred"] for c in cells]
    rr = [c["rough_ratio"] for c in cells]
    di = [c["delta_iso"] for c in cells]
    dsm = [c["delta_smooth"] for c in cells]

    f1 = spearmanr(vp, di)
    f2 = spearmanr(rr, dsm)
    none_u = float(np.mean([c["rmse_none"] for c in cells]))
    iso_u = float(np.mean([c["rmse_iso"] for c in cells]))
    d_unit = (iso_u - none_u) / none_u * 100

    verdict = {
        "n_cells": len(cells), "src_rough": src_rough, "r_src": R_SRC,
        "F1_rho": float(f1.statistic), "F1_p": float(f1.pvalue),
        "F1_holds": bool(f1.statistic < 0),
        "F2_rho": float(f2.statistic), "F2_p": float(f2.pvalue),
        "F2_holds": bool(f2.statistic < 0),
        "unit_delta_iso_pct": d_unit,
        "F3_holds": bool(-14.6 <= d_unit <= 5.0),
        "F4_holds": bool(d_unit < 11.3),
    }
    json.dump({"verdict": verdict, "cells": cells}, OUT.open("w"), indent=1)

    print(f"MATR-CLO, 9 cells, source roughness={src_rough:.5f}")
    print(f"F1 v_pred~delta_iso:      rho={f1.statistic:+.3f} p={f1.pvalue:.3f} → "
          + ("holds" if verdict['F1_holds'] else "falsified") + " (predicted rho<0)")
    print(f"F2 rough_ratio~delta_sm:  rho={f2.statistic:+.3f} p={f2.pvalue:.3f} → "
          + ("holds" if verdict['F2_holds'] else "falsified") + " (predicted rho<0)")
    print(f"F3 Tier-S gate (apply iso): unit-level delta={d_unit:+.2f}% -> "
          + ("holds" if verdict['F3_holds'] else "falsified") + " (interval [-14.6,+5.0])")
    print("F4 safe (<+11.3%): "
          + ("holds" if verdict['F4_holds'] else "falsified"))


if __name__ == "__main__":
    main()
