"""bexp7: the context-size ladder -- is the full context best?

Motivation: the frozen protocol uses a 2048-row context, but that is only 1.9 per
cent of the actual training rows on HUST and 3.0 per cent on MIT. The two units
where the history stack failed (HUST +1.6 per cent, TJU-2 +36.0 per cent) happen to
be the most heavily truncated ones, so truncation has to be ruled out as the cause.

Design: context in {2048, 8192, full} x {raw, history} x 3 seeds, reference
protocol. Mid-sized units run first; the full-context runs on HUST/MIT are
controlled separately.

Three possible readings:
  A the history stack turns positive with a larger context -> the earlier failures
    were an artefact of truncation
  B it stays negative -> "the history stack does not help when data are plentiful"
    is the real conclusion
  C both views improve equally -> context size and feature view are orthogonal
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.protocols import evaluate_unit

OUT = Path("results/battery/bexp7_context_ladder.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ordered by data size, smallest first, so results arrive early and a run can be
# interrupted at any point
DEFAULT_UNITS = ["XJTU-satellite", "TJU-3", "TJU-1", "TJU-2", "MIT", "HUST"]
DEFAULT_CAPS = [2048, 8192, 10 ** 9]     # 1e9 = the full context (no truncation)
SEEDS = [0, 1, 2]


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add((r["unit"], r["view"], r["cap"], r["seed"]))
            except json.JSONDecodeError:
                continue
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=DEFAULT_UNITS)
    ap.add_argument("--caps", nargs="+", type=int, default=DEFAULT_CAPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    done = done_keys()
    unit_map = {uid: (ds, bk) for uid, ds, bk in FIT_UNITS}
    for uid in args.units:
        ds, bk = unit_map[uid]
        unit = load_fit_unit(ds, bk)
        n_rows = sum(len(c.y) for c in unit.train_cells)
        print(f"\n=== {uid}: {n_rows} training rows ===", flush=True)
        for cap in args.caps:
            eff = min(cap, n_rows)
            for view in ("raw", "history"):
                for seed in args.seeds:
                    key = (uid, view, cap, seed)
                    if key in done:
                        continue
                    t0 = time.time()
                    r = evaluate_unit(unit, view, seed, backbone="tabpfn",
                                      context_cap=cap)
                    r["cap"] = cap
                    r["cap_effective"] = eff
                    r["n_train_rows_full"] = n_rows
                    del r["per_cell"]           # keep the file small
                    with OUT.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"  cap={cap if cap < 10**9 else 'FULL':>5} "
                          f"({eff:>6} rows) {view:<8} seed={seed} "
                          f"macro={r['rmse_cell_macro']:.6f} "
                          f"overall={r['rmse_overall']:.6f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    report()


def report():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    units = sorted({r["unit"] for r in rows},
                   key=lambda u: DEFAULT_UNITS.index(u) if u in DEFAULT_UNITS else 99)
    print("\n===== context-size ladder (cell_macro, mean over seeds) =====")
    print(f"{'unit':<16}{'context':>9}{'rows':>8}{'raw':>10}{'history':>10}"
          f"{'H-R Δ%':>9}")
    for uid in units:
        for cap in sorted({r["cap"] for r in rows if r["unit"] == uid}):
            m = {}
            eff = None
            for view in ("raw", "history"):
                v = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
                     and r["cap"] == cap and r["view"] == view]
                if v:
                    m[view] = float(np.mean(v))
                    eff = [r["cap_effective"] for r in rows if r["unit"] == uid
                           and r["cap"] == cap][0]
            if len(m) < 2:
                continue
            d = (m["history"] - m["raw"]) / m["raw"] * 100
            tag = "FULL" if cap >= 10 ** 9 else str(cap)
            print(f"{uid:<16}{tag:>8}{eff:>8}{m['raw']:>10.6f}"
                  f"{m['history']:>10.6f}{d:>+9.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
