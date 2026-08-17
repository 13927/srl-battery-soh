"""bexp13: frozen verification of the source-side rebound-ratio rule on the
unexposed library (MATR-CLO).

Frozen predictions:
  P1  iso    delta% in [-14.6, +5.0], point estimate -4.8%
  P2a causal delta% > 0 (global linear extrapolation)
  P2b causal delta% < 0 (local same-band analogy) -- contradicts P2a, the data
      decide
  P3  iso    delta% < +25% (a safety bound)

Run once, 3 seeds, no tuning. The frozen MATR-CLO features and imputation rule of
bexp5 are reused.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from battery_lab.physics_posthoc import POSTPROC, apply_postproc
from bexp5_unexposed_frozen import load_unit_npz, run_view

OUT = Path("results/battery/bexp13_recovery_rule_validation.json")
SEEDS = [0, 1, 2]
FROZEN = {"P1_iso_interval": [-14.6, 5.0], "P1_iso_point": -4.8,
          "P2a_causal_positive": True, "P2b_causal_negative": True,
          "P3_iso_safe_below": 25.0, "r_source": 0.0750}


def main():
    unit = load_unit_npz()
    print(f"MATR-CLO: source {len(unit.train_cells)} / "
          f"target {len(unit.test_cells)} cells",
          flush=True)
    # per seed: one inference, then every post-processing variant
    per_seed = {m: [] for m in POSTPROC}
    detail = []
    for seed in SEEDS:
        t0 = time.time()
        for view in ("history",):          # frozen: history view only
            _, pred, y_te, te_ids = run_view(unit, view, seed)
            row = {"seed": seed, "view": view}
            for mode in POSTPROC:
                p = apply_postproc(pred, te_ids, mode)
                pc = {c: float(np.sqrt(np.mean((y_te[te_ids == c]
                                               - p[te_ids == c]) ** 2)))
                      for c in np.unique(te_ids)}
                row[mode] = {
                    "rmse_overall": float(np.sqrt(np.mean((y_te - p) ** 2))),
                    "rmse_cell_macro": float(np.mean(list(pc.values()))),
                    "bias": float(np.mean(p - y_te)),
                }
                per_seed[mode].append(row[mode]["rmse_overall"])
            detail.append(row)
            print(f"  seed={seed} none={row['none']['rmse_overall']:.6f} "
                  f"iso={row['iso']['rmse_overall']:.6f} "
                  f"causal={row['causal']['rmse_overall']:.6f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    base = float(np.mean(per_seed["none"]))
    deltas = {m: (float(np.mean(v)) - base) / base * 100
              for m, v in per_seed.items()}

    iso_d, cau_d = deltas["iso"], deltas["causal"]
    verdict = {
        "r_source": FROZEN["r_source"],
        "n_source_cells": len(unit.train_cells),
        "n_target_cells": len(unit.test_cells),
        "none_rmse_overall": base,
        "deltas_pct": deltas,
        "P1_iso_in_interval": bool(FROZEN["P1_iso_interval"][0] <= iso_d
                                   <= FROZEN["P1_iso_interval"][1]),
        "P1_iso_actual": iso_d,
        "P2a_holds": bool(cau_d > 0),
        "P2b_holds": bool(cau_d < 0),
        "P2_causal_actual": cau_d,
        "P3_iso_safe": bool(iso_d < FROZEN["P3_iso_safe_below"]),
    }
    verdict["rule_extrapolates"] = bool(verdict["P1_iso_in_interval"]
                                        and verdict["P3_iso_safe"])
    out = {"frozen": FROZEN, "verdict": verdict, "per_seed": per_seed,
           "detail": detail}
    json.dump(out, OUT.open("w"), indent=1)

    print("\n===== frozen verdict =====")
    print(f"  source-side rebound ratio r = {FROZEN['r_source']:.4f} "
          f"(low-rebound band)")
    print(f"  none  RMSE = {base:.6f}")
    for m in POSTPROC:
        print(f"  {m:<12} Δ% = {deltas[m]:+7.2f}")
    print(f"\n  P1 (iso ∈ [{FROZEN['P1_iso_interval'][0]},"
          f"{FROZEN['P1_iso_interval'][1]}]): measured {iso_d:+.2f}% -> "
          + ("holds" if verdict['P1_iso_in_interval'] else "falsified"))
    print(f"  P2a (causal > 0): measured {cau_d:+.2f}% -> "
          + ("holds" if verdict['P2a_holds'] else "no"))
    print("  P2b (causal < 0): "
          + ("holds" if verdict['P2b_holds'] else "no"))
    print("  P3 (iso < +25%): "
          + ("holds" if verdict['P3_iso_safe'] else "falsified"))
    print("\n  the rule extrapolates: "
          + ("yes" if verdict['rule_extrapolates'] else "no"))


if __name__ == "__main__":
    main()
