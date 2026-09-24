"""bexp53: reproducible audit of the 11.3% noise floor (response to R1-m6).

Background (reviewer R1-m6 / revision plan section 4-D):
  The paper uses a "noise floor of 11.3%" as a **frozen constant** (Methods writes it explicitly as
  "obtained from a seed-only replication ... the mean of those twenty values (11.35%)"), but the
  repository contains **no reproducible algorithm** and **no source data file**. The reviewer asked
  for the number of pairs, the units, the context sampling modes, and whether the statistic is the
  median of seed-to-seed variation.

Reconstructed definition (the convention fixed by this script, matching the Methods text sentence
by sentence):
  - Source: `results/battery/bexp11_coverage.jsonl` (history feature set, cap=8192)
  - Grouping: 4 units x 5 context sampling modes = 20 groups, each with 3 seeds
  - Per-group statistic: the **relative range** (max-min)/mean of the 3 seeds' rmse_cell_macro
  - Noise floor = the **mean** of those 20 relative ranges
  - The paper text and Methods write "11.3%" / "(11.35%)"

================== decision criteria (recorded in this docstring; prior registration is not claimed -- see Section 2.7) ==================
D1: if the recomputed value falls inside **11.30 - 11.45 (%)**, the definition of that "frozen
    constant" is judged to hold, and the full 20-group detail is printed (unit / mode / 3 seeds /
    range%).
    -> holds: Methods may cite this script as the reproducible audit;
    -> fails: report the recomputed value as measured, and state the difference from the paper's
       constant.
Also: print one machine-readable summary line to the .out log, of the form
  NOISE_FLOOR=11.35 (groups=20, seeds=3, units=4, modes=5, statistic=mean_relative_range)
==================================================================

Note: **pure archive read**; no existing file under `results/battery/` is modified; only this
      script's summary JSON and .out log are added.

Run:
./.venv/bin/python experiments/bexp53_noise_floor_audit.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

SRC = ROOT / "results/battery/bexp11_coverage.jsonl"
OUT = ROOT / "results/battery/bexp53_noise_floor.json"
LO, HI = 11.30, 11.45          # criterion interval (%)


def main():
    rows = [json.loads(l) for l in SRC.read_text().splitlines()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["unit"], r["mode"])].append(r["rmse_cell_macro"])

    details = []
    for (unit, mode), vals in sorted(groups.items()):
        v = np.array(vals, dtype=float)
        rel_range = (v.max() - v.min()) / v.mean() * 100.0
        details.append({
            "unit": unit, "mode": mode, "seeds": sorted(
                r["seed"] for r in rows
                if r["unit"] == unit and r["mode"] == mode),
            "vals": [round(float(x), 9) for x in v],
            "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "rel_range_pct": float(rel_range),
        })

    stats = np.array([d["rel_range_pct"] for d in details])
    floor = float(stats.mean())
    units = sorted({d["unit"] for d in details})
    modes = sorted({d["mode"] for d in details})
    seeds = sorted({s for d in details for s in d["seeds"]})

    ok = LO <= floor <= HI
    summary = {
        "_meta": {
            "source": str(SRC.relative_to(ROOT)),
            "n_groups": len(details), "n_units": len(units),
            "n_modes": len(modes), "seeds_per_group": len({len(d["seeds"]) for d in details}),
            "seeds": seeds, "units": units, "modes": modes,
            "statistic": "mean_relative_range",
            "noise_floor_pct": floor,
            "criterion_range_pct": [LO, HI],
            "criterion_D1_ok": bool(ok),
            "note": "archive read only; no existing file under results/battery is modified",
        },
        "per_group": details,
    }
    json.dump(summary, OUT.open("w"), indent=1)

    # ---- stdout report ----
    print("=" * 92)
    print("bexp53 noise-floor audit (definition: 20 groups = 4 units x 5 modes, per-group relative range of rmse_cell_macro over 3 seeds, then the mean)")
    print("=" * 92)
    print(f"{'unit':<10}{'mode':<16}{'seeds':<12}{'mean':>12}{'min':>12}{'max':>12}{'range%':>9}")
    for d in details:
        print(f"{d['unit']:<10}{d['mode']:<16}"
              f"{','.join(str(s) for s in d['seeds']):<12}"
              f"{d['mean']:>12.6f}{d['min']:>12.6f}{d['max']:>12.6f}"
              f"{d['rel_range_pct']:>9.3f}")
    print("-" * 92)
    print(f"groups={len(details)}  units={len(units)}  modes={len(modes)}  seeds/group={len(seeds)}")
    print(f"noise floor = mean of the 20 group relative ranges = {floor:.4f}%  (paper constant 11.3% / 11.35%)")
    print(f"criterion D1 (interval [{LO}, {HI}]%): "
          f"{'holds' if ok else 'fails'}")
    print()
    print(f"NOISE_FLOOR={floor:.2f} (groups={len(details)}, "
          f"seeds={len(seeds)}, units={len(units)}, modes={len(modes)}, "
          f"statistic=mean_relative_range)")


if __name__ == "__main__":
    main()
