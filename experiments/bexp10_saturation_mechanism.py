"""bexp10: why the context saturates (H1 redundancy / H2 distant rows hurt).

H1: redundancy measures against the saturation point (defined from the bexp7 size
    curve)
H2: random / recent / early sampling at the same 8192 budget

Criteria, set in advance:
  H1 holds: redundancy correlates negatively with the gain from 8192 to a larger
            context (more redundancy, earlier saturation)
  H2 holds: recent is clearly better than random (distant rows do hurt)
  neither:  the mechanism stays open and the paper reports the phenomenon only
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from battery_lab.saturation_mechanism import redundancy_metrics, sampling_comparison

RED = Path("results/battery/bexp10_redundancy.json")
SAMP = Path("results/battery/bexp10_sampling.jsonl")
LADDER = Path("results/battery/bexp7_context_ladder.jsonl")
# the H2 contrast needs a pool larger than 8192, otherwise sampling is vacuous
H2_UNITS = ["TJU-1", "TJU-2", "MIT", "HUST"]
SEEDS = [0, 1, 2]


def run_h1():
    data = json.load(RED.open()) if RED.exists() else {}
    for uid, ds, bk in FIT_UNITS:
        if uid in data:
            continue
        unit = load_fit_unit(ds, bk)
        data[uid] = redundancy_metrics(unit)
        json.dump(data, RED.open("w"), indent=1)
        d = data[uid]
        print(f"H1 {uid:<15} eff_rank={d['effective_rank']:.2f}/{d['dim']} "
              f"({d['eff_rank_ratio']:.3f}) nbr/glob={d['neighbor_dist_ratio']:.4f} "
              f"dedup={d['dedup_ratio_1pct']:.3f} rows/cell={d['rows_per_cell']:.0f}",
              flush=True)
    return data


def run_h2():
    done = set()
    if SAMP.exists():
        for line in SAMP.read_text().splitlines():
            r = json.loads(line)
            done.add((r["unit"], r["mode"], r["seed"]))
    for uid in H2_UNITS:
        ds, bk = {u: (d, b) for u, d, b in FIT_UNITS}[uid]
        unit = load_fit_unit(ds, bk)
        for seed in SEEDS:
            todo = [m for m in ("random", "recent", "early")
                    if (uid, m, seed) not in done]
            if not todo:
                continue
            t0 = time.time()
            rows = sampling_comparison(unit, "history", seed, cap=8192,
                                       modes=tuple(todo))
            with SAMP.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                    print(f"H2 {uid:<8}{r['mode']:<8}seed={seed} "
                          f"macro={r['rmse_cell_macro']:.6f} "
                          f"(n_ctx={r['n_ctx']})", flush=True)
            print(f"   [{time.time()-t0:.0f}s]", flush=True)


def report():
    print("\n" + "=" * 70)
    print("H1: redundancy vs saturation (relative gain from 8192 to a larger "
          "context; negative = larger is better)")
    red = json.load(RED.open())
    lad = [json.loads(l) for l in LADDER.read_text().splitlines()]
    gains, xs_rank, xs_dedup, xs_nbr, names = [], [], [], [], []
    for uid in {r["unit"] for r in lad}:
        u = [r for r in lad if r["unit"] == uid and r["view"] == "raw"]
        caps = sorted({r["cap"] for r in u})
        if len(caps) < 2 or uid not in red:
            continue
        m8 = np.mean([r["rmse_cell_macro"] for r in u if r["cap"] == 8192])
        big = max(c for c in caps if c > 8192) if any(c > 8192 for c in caps) else None
        if big is None or not np.isfinite(m8):
            continue
        mb = np.mean([r["rmse_cell_macro"] for r in u if r["cap"] == big])
        g = (mb - m8) / m8 * 100          # negative = the larger context is better
        gains.append(g); names.append(uid)
        xs_rank.append(red[uid]["eff_rank_ratio"])
        xs_dedup.append(red[uid]["dedup_ratio_1pct"])
        xs_nbr.append(red[uid]["neighbor_dist_ratio"])
        print(f"  {uid:<15} 8192→{big}: {g:+.2f}%  "
              f"eff_rank_ratio={red[uid]['eff_rank_ratio']:.3f} "
              f"dedup={red[uid]['dedup_ratio_1pct']:.3f}")
    if len(gains) >= 4:
        for nm, xs in [("eff_rank_ratio", xs_rank), ("dedup_ratio", xs_dedup),
                       ("nbr_dist_ratio", xs_nbr)]:
            r = spearmanr(xs, gains)
            print(f"  correlation {nm:<16} vs gain: rho={r.statistic:+.3f} "
                  f"p={r.pvalue:.3f}")
        print("  criterion: if redundancy (low dedup / low effective rank) goes "
              "with earlier saturation (gain near zero or positive), H1 is "
              "supported")
    else:
        print(f"  too few samples (n={len(gains)}) for a correlation")

    print("\n" + "=" * 70)
    print("H2: sampling contrast at the same 8192 budget (history view, "
          "cell_macro)")
    if not SAMP.exists():
        print("  (no data)"); return
    rows = [json.loads(l) for l in SAMP.read_text().splitlines()]
    print(f"{'unit':<10}{'random':>10}{'recent':>10}{'early':>10}"
          f"{'rec-rand%':>11}{'early-rand%':>12}")
    pairs_rec, pairs_rand, pairs_early = [], [], []
    for uid in H2_UNITS:
        u = [r for r in rows if r["unit"] == uid]
        m = {}
        for mode in ("random", "recent", "early"):
            v = [r["rmse_cell_macro"] for r in u if r["mode"] == mode]
            m[mode] = float(np.mean(v)) if v else np.nan
        if not np.isfinite(m["random"]):
            continue
        dr = (m["recent"] - m["random"]) / m["random"] * 100
        de = (m["early"] - m["random"]) / m["random"] * 100
        print(f"{uid:<10}{m['random']:>10.6f}{m['recent']:>10.6f}"
              f"{m['early']:>10.6f}{dr:>+11.2f}{de:>+12.2f}")
        for seed in SEEDS:
            rr = [r["rmse_cell_macro"] for r in u if r["mode"] == "random" and r["seed"] == seed]
            rc = [r["rmse_cell_macro"] for r in u if r["mode"] == "recent" and r["seed"] == seed]
            re_ = [r["rmse_cell_macro"] for r in u if r["mode"] == "early" and r["seed"] == seed]
            if rr and rc and re_:
                pairs_rand.append(rr[0]); pairs_rec.append(rc[0]); pairs_early.append(re_[0])
    if len(pairs_rand) >= 5:
        for nm, arr in [("recent", pairs_rec), ("early", pairs_early)]:
            d = np.array(arr) - np.array(pairs_rand)
            try:
                p = wilcoxon(arr, pairs_rand).pvalue
            except ValueError:
                p = 1.0
            print(f"  {nm} vs random: mean delta={d.mean()*1000:+.3f} (x1e-3) "
                  f"win={np.mean(d<0):.0%} p={p:.3f}")
        print("  criterion: recent clearly better than random supports H2 "
              "(distant rows hurt); otherwise H2 fails")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        run_h1()
        run_h2()
        report()
