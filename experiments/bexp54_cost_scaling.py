"""bexp54: convert timing to fleet scale (per 10,000 cells/day) energy and CO2e -- response to R1-m17 / R1-M10.

Background (reviewers R1-m17 / R1-M10 / revision plan Sections 4-E, 4-F):
  The paper text states "GBDT ~3 s/unit, TabPFN 10-100 s/unit", but (a) it does **not
  record the CPU model used for timing**, and (b) it does **not convert timing to fleet
  scale** energy/carbon. This script **measures** GBDT fit/predict timing on this machine
  and records the hardware identifier; the TabPFN **weights are missing locally and cannot
  be re-measured**, so its timing is taken from the archive (explicitly flagged as archive values).

Local measurement (GBDT):
  Protocol frozen identically to bexp14 / bexp42: history 81 dims, author_minmax
  normalization, GBDT (HistGradientBoostingRegressor defaults), context cap=2048.
  Per unit x 3 seeds, take the median. Small-sample HistGBDT was observed to be **slower
  under multiple threads** (thread overhead > parallel gain), so two readings are given:
  default (library default threads) and single (threadpoolctl limited to 1 thread,
  wall ~ CPU core-seconds).

Archive (TabPFN, **not re-measured locally**):
  - `results/battery/bexp1_run.out` (run log, per-line "(Ns)" integer seconds, cell counts
    taken from `bexp1_author_protocol.jsonl`)
  - `results/battery/bexp7_context_ladder.jsonl` (runtime_s)
  Both are flagged as archive timing and do not represent this machine's hardware.

Conversion (cost quantity = CPU seconds per cell per refresh):
  Two cost readings (each divides per unit by that unit's cell count, then takes the median over 11 units):
    (R1) amortized  : (fit + predict) / n_cells      # assumes the model is retrained on every refresh (conservative upper bound)
    (R2) inference  : predict / n_cells              # assumes the model is already trained, refresh = inference (lower bound)
  per 10,000 cells/day:
    CPU seconds/day = 10000 * per-cell cost (s);  CPU hours/day = CPU seconds/day / 3600
  Energy (parameterized, **not measured power**):
    Wh/day = CPU hours/day * P_active(W);  CO2e(g/day) = Wh/day * CI(kg CO2e/kWh)
  Assumptions: P_active in {15, 45} W (values within the 10-65 W active range of a
        typical desktop/laptop CPU);
        CI in {0.10, 0.30, 0.50} kg CO2e/kWh (low/mid/high-carbon grid scenarios).
  **All assumptions are listed explicitly; the output is a parameterized estimate and
  must not be written as "measured energy".**

================== decision criteria (recorded in this docstring; prior registration is not claimed -- see Section 2.7) ==================
E1: If the measured GBDT per-cell refresh cost <= 0.01 s,
    then the fleet-level analysis overhead of the GBDT route is judged to be at the
    **order of 1 CPU-hour/10k cells/day or below**;
    otherwise report as measured (and give the actual CPU-hours/10k cells/day).
    This script gives a criterion verdict separately for the (R1) and (R2) readings.
==================================================================

Run:
./.venv/bin/python experiments/bexp54_cost_scaling.py
"""

import json
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, subsample_context
from battery_lab.temporal_features import build_view

OUT = ROOT / "results/battery/bexp54_cost_scaling.json"
SEEDS = [0, 1, 2]                # per unit, median over 3 seeds
CAP = CONTEXT_CAP               # 2048
FLEET_CELLS = 10_000
POWERS_W = [15, 45]             # parameterized active-power levels (typical desktop CPU)
CARBON_CI = {"low_0.10": 0.10, "mid_0.30": 0.30, "high_0.50": 0.50}  # kg CO2e/kWh


def hw_info():
    """This machine's hardware identifier (the CPU model required by R1-M10)."""
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except Exception as e:  # noqa: BLE001
            return f"<unavailable: {e}>"
    return {
        "cpu_brand": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "model": sh(["sysctl", "-n", "hw.model"]),
        "ncpu": sh(["sysctl", "-n", "hw.ncpu"]),
        "memsize_bytes": sh(["sysctl", "-n", "hw.memsize"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def assemble(cells, Xn):
    xs, ys = [], []
    for c in cells:
        xs.append(build_view(Xn[c.cell_id], "history", cycle_mode="author_minmax"))
        ys.append(c.y)
    return np.vstack(xs), np.concatenate(ys)


def time_mode(X_tr, y_tr, X_te, threads):
    """Measure GBDT (fit/predict) timing under a given thread count, median over 3 seeds."""
    ctx = threadpool_limits(limits=1) if threads == 1 else nullcontext()
    with ctx:
        # warm-up (under this thread mode)
        HistGradientBoostingRegressor(random_state=0).fit(
            np.random.rand(256, 81), np.random.rand(256))
        fits, preds = [], []
        for s in SEEDS:
            Xc, yc = subsample_context(X_tr, y_tr, CAP, s)
            t0 = time.perf_counter()
            m = HistGradientBoostingRegressor(random_state=s).fit(Xc, yc)
            fits.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            m.predict(X_te)
            preds.append(time.perf_counter() - t0)
    return {"fit_s": float(np.median(fits)), "predict_s": float(np.median(preds)),
            "fit_s_seeds": [round(x, 6) for x in fits],
            "predict_s_seeds": [round(x, 6) for x in preds]}


def measure_gbdt():
    """Measure GBDT timing per unit: two readings, default (library default threads) and single (1 thread)."""
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    recs = {}
    for uid, _, _ in FIT_UNITS:
        ds, bk = umap[uid]
        unit = load_fit_unit(ds, bk)
        Xn = normalize_cells(unit.cells, "author_minmax")
        X_tr, y_tr = assemble(unit.train_cells, Xn)
        X_te, y_te = assemble(unit.test_cells, Xn)
        n_cells = len(unit.cells)
        d = time_mode(X_tr, y_tr, X_te, threads=None)
        s1 = time_mode(X_tr, y_tr, X_te, threads=1)
        rec = {"n_cells": n_cells, "n_train_rows_ctx": int(len(y_tr)),
               "n_test_rows": int(len(y_te)), "default": d, "single": s1}
        for tag in ("default", "single"):
            r = rec[tag]
            r["amortized_per_cell_s"] = (r["fit_s"] + r["predict_s"]) / n_cells
            r["inference_per_cell_s"] = r["predict_s"] / n_cells
            r["predict_per_row_s"] = r["predict_s"] / max(len(y_te), 1)
        recs[uid] = rec
        print(f"  [GBDT] {uid:<15} cells={n_cells:<4} "
              f"default fit={d['fit_s']:.3f} pred={d['predict_s']:.4f} am/cell={d['amortized_per_cell_s']:.5f} | "
              f"1-thr fit={s1['fit_s']:.3f} pred={s1['predict_s']:.4f} am/cell={s1['amortized_per_cell_s']:.5f}",
              flush=True)
    return recs


def archived_tabpfn():
    """Archive TabPFN timing (**not re-measured locally**). Returns per-unit per-cell cost and source note."""
    out = {}
    # (1) bexp7_context_ladder.jsonl: runtime_s (history)
    p7 = ROOT / "results/battery/bexp7_context_ladder.jsonl"
    if p7.exists():
        rows = [json.loads(l) for l in p7.read_text().splitlines()]
        for uid in [u for u, _, _ in FIT_UNITS]:
            sel = [r for r in rows if r["unit"] == uid and r["view"] == "history"]
            if not sel:
                continue
            fav = [r for r in sel if r.get("cap") == CAP] or sel
            rt = float(np.median([r["runtime_s"] for r in fav]))
            ncell = fav[0]["n_train_cells"] + fav[0]["n_test_cells"]
            out.setdefault(uid, {})["bexp7_history"] = {
                "runtime_s_median": rt, "n_cells": ncell,
                "per_cell_cost_s": rt / ncell, "cap": fav[0].get("cap"),
                "source": "archived (bexp7_context_ladder.jsonl runtime_s)",
            }
    # (2) bexp1_run.out: per-line "(Ns)" integer seconds (history, tabpfn); cell counts taken from bexp1 jsonl
    cellcount = {}
    p1j = ROOT / "results/battery/bexp1_author_protocol.jsonl"
    if p1j.exists():
        for l in p1j.read_text().splitlines():
            r = json.loads(l)
            if r.get("backbone") == "tabpfn" and r.get("view") == "history":
                cellcount[r["unit"]] = r["n_train_cells"] + r["n_test_cells"]
    p1 = ROOT / "results/battery/bexp1_run.out"
    if p1.exists():
        per_unit = {}
        for line in p1.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4 and "tabpfn" in line and "history" in line \
                    and parts[-1].startswith("(") and parts[-1].endswith("s)"):
                per_unit.setdefault(parts[0], []).append(float(parts[-1].strip("()s")))
        for uid, secs in per_unit.items():
            rt = float(np.median(secs))
            ncell = cellcount.get(uid)
            rec = {"runtime_s_median": rt, "runtime_s_list": secs,
                   "source": "archived (bexp1_run.out integer seconds)"}
            if ncell:
                rec["n_cells"] = ncell
                rec["per_cell_cost_s"] = rt / ncell
            out.setdefault(uid, {})["bexp1_run_out"] = rec
    return out


def fleet(cost_s):
    """Per-cell cost (s) -> CPU-h, Wh, g CO2e per 10,000 cells/day."""
    cpu_s = cost_s * FLEET_CELLS
    cpu_h = cpu_s / 3600.0
    rows = []
    for w in POWERS_W:
        wh = cpu_h * w
        for name, ci in CARBON_CI.items():
            rows.append({"power_W": w, "ci_scenario": name, "ci_kg_per_kWh": ci,
                         "cpu_h_per_day": cpu_h, "wh_per_day": wh, "gCO2e_per_day": wh * ci})
    return {"cpu_s_per_day": cpu_s, "cpu_h_per_day": cpu_h, "scenarios": rows}


def summarize_costs(gbdt, tag):
    """For a given thread reading, aggregate per-unit costs -> median cost reading."""
    amort = np.array([r[tag]["amortized_per_cell_s"] for r in gbdt.values()])
    infer = np.array([r[tag]["inference_per_cell_s"] for r in gbdt.values()])
    return {"amortized_per_cell_s_median": float(np.median(amort)),
            "inference_per_cell_s_median": float(np.median(infer))}


def main():
    hw = hw_info()
    print("=" * 96)
    print("bexp54 timing -> fleet (per 10,000 cells/day) energy and CO2e -- parameterized estimate (not measured power)")
    print("=" * 96)
    print(f"Hardware: {hw['cpu_brand']} | {hw['model']} | ncpu={hw['ncpu']} "
          f"| mem={int(hw['memsize_bytes'])/1e9:.1f} GB | {hw['platform']}")
    print(f"Python {hw['python']}; protocol: history 81 dims, author_minmax, GBDT, cap={CAP}")

    print("\n[locally measured GBDT]")
    gbdt = measure_gbdt()

    print("\n[archived TabPFN -- not re-measured locally]")
    tab = archived_tabpfn()
    tab_vals = []
    for uid, d in tab.items():
        for k, v in d.items():
            if "per_cell_cost_s" in v:
                tab_vals.append(v["per_cell_cost_s"])
                print(f"  [archive] {uid:<15}{k:<18} runtime={v['runtime_s_median']:.1f}s "
                      f"cells={v['n_cells']:<4} per_cell={v['per_cell_cost_s']:.4f}s")
    tab_median = float(np.median(tab_vals)) if tab_vals else float("nan")

    costs = {"default": summarize_costs(gbdt, "default"),
             "single": summarize_costs(gbdt, "single")}
    print("\n[GBDT cost readings (per unit / cell count, median over 11 units)]")
    for tag in ("default", "single"):
        c = costs[tag]
        print(f"  {tag:<8} (R1)amortized={c['amortized_per_cell_s_median']:.5f}s  "
              f"(R2)inference={c['inference_per_cell_s_median']:.6f}s")

    # criterion E1: primary reading is single (~CPU core-seconds) + amortized; all readings are listed as well
    verdicts = {}
    for tag in ("default", "single"):
        for read in ("amortized", "inference"):
            per_cell = costs[tag][f"{read}_per_cell_s_median"]
            fl = fleet(per_cell)
            verdicts[f"{tag}_{read}"] = {
                "per_cell_s": per_cell, "ok": bool(per_cell <= 0.01),
                "cpu_h_per_10k_day": fl["cpu_h_per_day"]}
    print(f"\n=== frozen verdict E1 (threshold 0.01 s/cell) ===")
    for k, v in verdicts.items():
        print(f"  {k:<20} per_cell={v['per_cell_s']:.6f}s "
              f"{'<=' if v['ok'] else '>'} 0.01 -> {'holds ✓' if v['ok'] else 'fails ✗'} "
              f"(fleet={v['cpu_h_per_10k_day']:.4f} CPU-h/10k cells/day)")
    primary = verdicts["single_amortized"]
    print(f"  primary reading (single_amortized): {'holds ✓' if primary['ok'] else 'fails ✗ report as measured'}"
          f"  actual ~ {primary['cpu_h_per_10k_day']:.4f} CPU-h/10k cells/day"
          f" (still {'below' if primary['cpu_h_per_10k_day']<1 else 'above'} 1 CPU-h)")

    gbdt_fleet = fleet(primary["per_cell_s"])
    tab_fleet = fleet(tab_median) if np.isfinite(tab_median) else None

    print(f"\n--- per {FLEET_CELLS} cells/day (parameterized estimate, not measured energy; primary reading single_amortized) ---")
    print(f"GBDT   per cell {primary['per_cell_s']:.5f}s -> CPU {gbdt_fleet['cpu_h_per_day']:.4f} h/day")
    for r in gbdt_fleet["scenarios"]:
        print(f"        P={r['power_W']:>2} W, CI={r['ci_scenario']:<8} "
              f"-> {r['wh_per_day']:8.2f} Wh/day, {r['gCO2e_per_day']:9.3f} g CO2e/day")
    if tab_fleet:
        print(f"TabPFN(archive) per cell {tab_median:.4f}s -> CPU {tab_fleet['cpu_h_per_day']:.4f} h/day")
        for r in tab_fleet["scenarios"]:
            print(f"        P={r['power_W']:>2} W, CI={r['ci_scenario']:<8} "
                  f"-> {r['wh_per_day']:8.2f} Wh/day, {r['gCO2e_per_day']:9.3f} g CO2e/day")

    print("\n[assumption list (explicit)]")
    print(f"  * fleet scale = {FLEET_CELLS} cells/day, one refresh per cell per day")
    print("  * GBDT cost readings divide per unit by that unit's cell count, then take the median over 11 units; locally measured, median over 3 seeds")
    print("  * (R1)amortized=(fit+predict)/cells  (R2)inference=predict/cells")
    print("  * TabPFN cost = archive runtime_s / unit cell count; **archive value, not re-measured locally**")
    print(f"  * CPU active-power level P = {POWERS_W} W (values within the 10-65 W active range of a typical desktop CPU)")
    print(f"  * grid carbon intensity CI = {list(CARBON_CI.values())} kg CO2e/kWh (low/mid/high-carbon scenarios)")
    print("  * energy and CO2e are **parameterized estimates**, not measured power")

    print()
    print(f"COST_SCALING(hw={hw['cpu_brand']}) "
          f"gbdt_single_amortized_per_cell_s={primary['per_cell_s']:.5f} "
          f"gbdt_cpu_h_per_10k_day={primary['cpu_h_per_10k_day']:.4f} "
          f"gbdt_default_amortized_per_cell_s={costs['default']['amortized_per_cell_s_median']:.5f} "
          f"gbdt_single_inference_per_cell_s={costs['single']['inference_per_cell_s_median']:.6f} "
          f"tabpfn_archived_per_cell_s={tab_median:.4f} "
          f"tabpfn_cpu_h_per_10k_day={(tab_fleet['cpu_h_per_day'] if tab_fleet else float('nan')):.4f} "
          f"criterion_E1_primary_ok={primary['ok']}")

    json.dump({
        "_meta": {
            "hardware": hw,
            "protocol": {"view": "history", "features": 81, "normalize": "author_minmax",
                         "backbone": "HistGradientBoostingRegressor(cap=%d)" % CAP},
            "seeds": SEEDS, "fleet_cells": FLEET_CELLS, "cap": CAP,
            "assumptions": {"power_W": POWERS_W, "carbon_ci_kg_per_kWh": CARBON_CI,
                            "note": "energy and CO2e are parameterized estimates, not measured power; TabPFN timing comes from the archive; "
                                    "R1=amortized(fit+predict)/cells, R2=inference=predict/cells"},
            "cost_readings": costs,
            "criterion_E1": {"threshold_per_cell_s": 0.01, "verdicts": verdicts,
                             "primary": "single_amortized", "primary_ok": bool(primary["ok"])},
        },
        "gbdt_measured": gbdt, "gbdt_fleet": gbdt_fleet,
        "tabpfn_archived": tab, "tabpfn_per_cell_s_median": tab_median,
        "tabpfn_fleet": tab_fleet,
    }, OUT.open("w"), indent=1)
    print(f"\n[written to disk] {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
