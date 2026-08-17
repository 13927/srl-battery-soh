"""dataset_summary: build the archive behind the dataset table of Appendix A (so
check_provenance can verify it).

Section 2.1 states cell counts and splits only; chemistry, nominal capacity and
cycle ranges are what a battery reader expects to see. This script writes every
number of that table into an archive, so it can be recomputed like the results of
every other experiment.

The numbers come from two sources, neither of which is a hand-entered result:
  computed:  cell counts, held-out cell counts, per-cell cycle ranges and the
             lowest observed SOH -- obtained by actually loading the data through
             battery_lab.data_adapters under the official splits
  metadata:  chemistry, nominal capacity, operating-condition labels -- taken from
             the reference release (the TJU batch directories carry NCA/NCM in
             their names; TJU_NOMINAL and XJTU 2.0 / HUST 1.1 / MIT 1.1 are in
             data_adapters), while the TJU temperatures and rates are parsed from
             its cell ids (CY{temperature}-{charge}_{discharge}-#{index})

Run:    ./.venv/bin/python experiments/dataset_summary.py
Output: results/battery/dataset_summary.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battery_lab.data_adapters import (  # noqa: E402
    DATA_ROOT,
    FIT_UNITS,
    TJU_BATCHES,
    load_fit_unit,
)

OUT = ROOT / "results" / "battery" / "dataset_summary.json"

# chemistry and nominal capacity: metadata of the reference release (see the
# module docstring)
META = {
    "XJTU-2C": ("NCM", 2.0), "XJTU-3C": ("NCM", 2.0),
    "XJTU-R2.5": ("NCM", 2.0), "XJTU-R3": ("NCM", 2.0),
    "XJTU-RW": ("NCM", 2.0), "XJTU-satellite": ("NCM", 2.0),
    "TJU-1": ("NCA", 3.5), "TJU-2": ("NCM", 3.5), "TJU-3": ("NCM+NCA", 2.5),
    "HUST": ("LFP", 1.1), "MIT": ("LFP", 1.1),
}


def tju_conditions():
    """Parse temperature and charge/discharge rate from the TJU cell ids:
    CY{T}-{chg}_{dch}-#{n}."""
    out = {}
    for i, b in enumerate(TJU_BATCHES):
        temps, rates = set(), set()
        for f in sorted((DATA_ROOT / "TJU data" / b).glob("*.csv")):
            m = re.match(r"CY(\d+)-(\d+)_(\d+)", f.stem)
            if m:
                temps.add(int(m.group(1)))
                rates.add(f"{int(m.group(2)) / 10:g}C/{m.group(3)}C")
        out[f"TJU-{i + 1}"] = {"temperatures_C": sorted(temps),
                              "charge_discharge_rates": sorted(rates)}
    return out


def main():
    rows = {}
    for uid, ds, bk in FIT_UNITS:
        u = load_fit_unit(ds, bk)
        ncyc = [len(c.y) for c in u.cells]
        chem, nominal = META[uid]
        rows[uid] = {
            "chemistry": chem,
            "nominal_capacity_Ah": nominal,
            "cells": len(u.cells),
            "held_out_cells": sum(c.is_test for c in u.cells),
            "cycles_min": min(ncyc),
            "cycles_max": max(ncyc),
            "soh_min": round(min(float(c.y.min()) for c in u.cells), 4),
        }
        print(f"{uid:<16}{chem:<9}{nominal:>5} Ah  cells={rows[uid]['cells']:>4}"
              f"({rows[uid]['held_out_cells']:>3} held out)"
              f"  cycles={rows[uid]['cycles_min']}-{rows[uid]['cycles_max']}"
              f"  SOHmin={rows[uid]['soh_min']:.3f}", flush=True)

    state = {"units": rows, "tju_conditions": tju_conditions(),
             "totals": {"cells": sum(r["cells"] for r in rows.values()),
                        "held_out_cells": sum(r["held_out_cells"]
                                              for r in rows.values()),
                        "chemistries": dict(Counter(r["chemistry"]
                                                    for r in rows.values()))}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(state, OUT.open("w"), indent=1)
    print(f"\n{state['totals']}\n-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
