"""bexp1: alignment check plus the interaction matrix.

11 evaluation units x {raw, history} x {tabpfn, mlp} x 5 seeds, reference
protocol.
Output: results/battery/bexp1_author_protocol.jsonl (appended incrementally, so a
run can resume)

Acceptance gate: the four-library aggregate of TabPFN + history deviates by <=5
per cent from the previously reported numbers
  HUST 0.009070 / MIT 0.007372 / TJU 0.009676 / XJTU 0.007530
  (XJTU uses cell-macro, the others the reference sample-weighted overall; the
   three TJU batches are pooled by test-row count, the six XJTU batches by macro
   average over cells)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import evaluate_unit

OUT = Path("results/battery/bexp1_author_protocol.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
VIEWS = ["raw", "history"]
BACKBONES = ["tabpfn", "mlp"]

USER_REPORTED = {"HUST": 0.009070, "MIT": 0.007372,
                 "TJU": 0.009676, "XJTU": 0.007530}


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add((r["unit"], r["view"], r["backbone"], r["seed"]))
            except json.JSONDecodeError:
                continue
    return keys


def main():
    done = done_keys()
    units = {}
    for uid, ds, bk in FIT_UNITS:
        units[uid] = load_fit_unit(ds, bk)

    for uid, unit in units.items():
        for view in VIEWS:
            for backbone in BACKBONES:
                for seed in SEEDS:
                    key = (uid, view, backbone, seed)
                    if key in done:
                        continue
                    r = evaluate_unit(unit, view, seed, backbone=backbone)
                    with OUT.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"{uid:<14} {view:<8} {backbone:<7} seed={seed} "
                          f"overall={r['rmse_overall']:.6f} "
                          f"macro={r['rmse_cell_macro']:.6f} "
                          f"({r['runtime_s']:.0f}s)", flush=True)

    summarize()


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    print("\n===== acceptance gate: TabPFN + history, four-library aggregate "
          "vs previously reported =====")
    for lib, ref in USER_REPORTED.items():
        rs = [r for r in rows if r["unit"].startswith(lib)
              and r["view"] == "history" and r["backbone"] == "tabpfn"]
        if not rs:
            continue
        if lib == "XJTU":  # macro average over the six batches (macro per unit,
                           # averaged over seeds, then over units)
            unit_means = {}
            for r in rs:
                unit_means.setdefault(r["unit"], []).append(r["rmse_cell_macro"])
            val = float(np.mean([np.mean(v) for v in unit_means.values()]))
        elif lib == "TJU":  # the three batches pooled by test-row count
            unit_stats = {}
            for r in rs:
                unit_stats.setdefault(r["unit"], []).append(
                    (r["rmse_overall"], r["n_test_rows"]))
            vals, ws = [], []
            for u, lst in unit_stats.items():
                vals.append(np.mean([x[0] for x in lst]))
                ws.append(lst[0][1])
            val = float(np.average(vals, weights=ws))
        else:  # HUST/MIT are single units
            val = float(np.mean([r["rmse_overall"] for r in rs]))
        dev = (val - ref) / ref * 100
        flag = "PASS" if abs(dev) <= 5 else "FAIL"
        print(f"  {lib:<5} reproduced={val:.6f} reported={ref:.6f} "
              f"deviation={dev:+.2f}%  [{flag}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarize()
    else:
        main()
