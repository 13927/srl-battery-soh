"""bexp11: coverage criteria against random selection -- the last remaining lead.

Hypothesis H3: what determines context quality is coverage (label range,
        degradation stage, cell diversity) rather than similarity (kNN and
        clustering, all of which failed).

Criterion, binary and set in advance:
  some coverage strategy beats random in >= 3 of 4 units with a paired Wilcoxon
  p < 0.05
    -> H3 holds and a usable regularity exists
  otherwise
    -> random really is best, and "context selection offers no usable regularity
       here" becomes the settled conclusion
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from battery_lab.coverage_sampling import coverage_comparison
from battery_lab.data_adapters import FIT_UNITS, load_fit_unit

OUT = Path("results/battery/bexp11_coverage.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
# units whose pool exceeds 8192 (otherwise sampling is vacuous)
UNITS = ["TJU-1", "TJU-2", "MIT", "HUST"]
MODES = ("random", "soh_strat", "stage_strat", "cell_balanced", "soh_cell")
SEEDS = [0, 1, 2]


def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            keys.add((r["unit"], r["mode"], r["seed"]))
    return keys


def main():
    done = done_keys()
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    for uid in UNITS:
        ds, bk = umap[uid]
        unit = load_fit_unit(ds, bk)
        for seed in SEEDS:
            todo = tuple(m for m in MODES if (uid, m, seed) not in done)
            if not todo:
                continue
            t0 = time.time()
            rows = coverage_comparison(unit, "history", seed, cap=8192,
                                       modes=todo)
            with OUT.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                    print(f"{uid:<8}{r['mode']:<15}seed={seed} "
                          f"macro={r['rmse_cell_macro']:.6f} "
                          f"sohW={r['soh_wasserstein']:.4f} "
                          f"cellcov={r['cell_coverage']:.2f}", flush=True)
            print(f"   [{time.time()-t0:.0f}s]", flush=True)
    report()


def report():
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    print("\n" + "=" * 78)
    print("coverage strategies vs random (history, cell_macro, mean of 3 seeds)")
    hdr = f"{'unit':<9}" + "".join(f"{m[:11]:>12}" for m in MODES)
    print(hdr)
    for uid in UNITS:
        u = [r for r in rows if r["unit"] == uid]
        if not u:
            continue
        line = f"{uid:<9}"
        for m in MODES:
            v = [r["rmse_cell_macro"] for r in u if r["mode"] == m]
            line += f"{np.mean(v):>12.6f}" if v else f"{'-':>12}"
        print(line)

    print(f"\n{'unit':<9}" + "".join(f"{m[:9]+' d%':>12}" for m in MODES[1:]))
    for uid in UNITS:
        u = [r for r in rows if r["unit"] == uid]
        rnd = [r["rmse_cell_macro"] for r in u if r["mode"] == "random"]
        if not rnd:
            continue
        base = np.mean(rnd)
        line = f"{uid:<9}"
        for m in MODES[1:]:
            v = [r["rmse_cell_macro"] for r in u if r["mode"] == m]
            line += f"{(np.mean(v)-base)/base*100:>+12.2f}" if v else f"{'-':>12}"
        print(line)

    print("\n--- paired tests (paired by unit and seed; negative delta favours "
          "the coverage strategy) ---")
    verdict = {}
    for m in MODES[1:]:
        pa, pb = [], []
        for uid in UNITS:
            for s in SEEDS:
                a = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
                     and r["mode"] == m and r["seed"] == s]
                b = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid
                     and r["mode"] == "random" and r["seed"] == s]
                if a and b:
                    pa.append(a[0]); pb.append(b[0])
        if len(pa) < 5:
            continue
        d = np.array(pa) - np.array(pb)
        try:
            p = wilcoxon(pa, pb).pvalue
        except ValueError:
            p = 1.0
        n_units_win = 0
        for uid in UNITS:
            av = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid and r["mode"] == m]
            bv = [r["rmse_cell_macro"] for r in rows if r["unit"] == uid and r["mode"] == "random"]
            if av and bv and np.mean(av) < np.mean(bv):
                n_units_win += 1
        verdict[m] = (d.mean(), np.mean(d < 0), p, n_units_win)
        print(f"  {m:<15} mean delta={d.mean()*1000:+.3f} (x1e-3) "
              f"win={np.mean(d<0):.0%} p={p:.3f} "
              f"units won={n_units_win}/{len(UNITS)}")

    print("\n=== verdict ===")
    winners = [m for m, (dm, w, p, nu) in verdict.items()
               if nu >= 3 and p < 0.05 and dm < 0]
    if winners:
        print(f"  H3 holds: {winners} won >= 3/4 units with p<0.05")
        print("  -> a regularity exists: coverage beats random")
    else:
        print("  H3 fails: no coverage strategy met the criterion")
        print("  -> random really is best; context selection offers no usable "
              "regularity on this task")

    # association between the coverage diagnostics and performance
    print("\n--- coverage diagnostics vs performance (across strategies within a "
          "unit, testing the mechanism) ---")
    for uid in UNITS:
        u = [r for r in rows if r["unit"] == uid]
        if len(u) < 5:
            continue
        for key in ("soh_wasserstein", "cell_coverage", "stage_std"):
            xs = [r[key] for r in u]
            ys = [r["rmse_cell_macro"] for r in u]
            rho = spearmanr(xs, ys)
            if abs(rho.statistic) > 0.5 and rho.pvalue < 0.1:
                print(f"  {uid:<9}{key:<18} rho={rho.statistic:+.3f} p={rho.pvalue:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
