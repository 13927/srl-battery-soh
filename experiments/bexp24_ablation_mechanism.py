"""bexp24: (A) ablation over the four transform families, (B) the cell-level
regularity behind the gain.

A. Ablation (GBDT, reference protocol, cap=2048, 5 seeds, 11 units):
   conditions: raw / raw+one family (x4) / history minus one family (x4) / full
   history = 10 conditions
   purpose: the standard ablation table -- what each transform family is worth.
B. Mechanism: the per-cell history gain (bexp1 per_cell, TabPFN) against that
   cell's actual degradation rate (end-to-end label difference over cycles; using
   labels for analysis is legal), n of about 100+ cells.
   purpose: turn the within-batch observation on TJU-1 (the fastest-fading cells
   gain most) into a quantitative regularity across libraries.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import CONTEXT_CAP, fit_predict_gbdt, subsample_context
from battery_lab.temporal_features import cycle_author_minmax

OUT_A = Path("results/battery/bexp24_ablation.json")
OUT_B = Path("results/battery/bexp24_mechanism.json")
SEEDS = [0, 1, 2, 3, 4]
FAMILIES = ["lag1", "diff1", "from_first", "from_roll5"]


def family_blocks(Z):
    """The four families, computed exactly as in add_history_features."""
    n, d = Z.shape
    lag1 = np.vstack([Z[:1], Z[:-1]])
    diff1 = Z - lag1
    from_first = Z - Z[:1]
    csum = np.cumsum(Z, axis=0)
    roll5 = np.empty_like(Z)
    for t in range(n):
        lo = max(0, t - 4)
        total = csum[t] - (csum[lo - 1] if lo > 0 else 0)
        roll5[t] = total / (t - lo + 1)
    return {"lag1": lag1, "diff1": diff1, "from_first": from_first,
            "from_roll5": Z - roll5}


def assemble(cells, Xn, fams):
    xs, ys = [], []
    for c in cells:
        Z = Xn[c.cell_id]
        cyc = cycle_author_minmax(len(Z))
        blocks = family_blocks(Z)
        cols = [Z, cyc] + [blocks[f] for f in fams]
        xs.append(np.hstack(cols))
        ys.append(c.y)
    return np.vstack(xs), np.concatenate(ys)


def run_ablation():
    conds = ([("raw", [])] + [(f"+{f}", [f]) for f in FAMILIES]
             + [(f"-{f}", [g for g in FAMILIES if g != f]) for f in FAMILIES]
             + [("history", FAMILIES)])
    state = json.load(OUT_A.open()) if OUT_A.exists() else {}
    for uid, ds, bk in FIT_UNITS:
        state.setdefault(uid, {})
        unit = None
        for name, fams in conds:
            if name in state[uid]:
                continue
            if unit is None:
                unit = load_fit_unit(ds, bk)
                Xn = normalize_cells(unit.train_cells + unit.test_cells,
                                     "author_minmax")
                X_te, y_te = assemble(unit.test_cells, Xn, FAMILIES)  # placeholder
            # assemble each condition separately, test side included
            X_tr, y_tr = assemble(unit.train_cells, Xn, fams)
            X_t, y_t = assemble(unit.test_cells, Xn, fams)
            vals = []
            for seed in SEEDS:
                Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
                pred = fit_predict_gbdt(Xc, yc, X_t, seed)
                vals.append(float(np.sqrt(np.mean((y_t - pred) ** 2))))
            state[uid][name] = {"rmse_overall": float(np.mean(vals)),
                                "seed_vals": vals, "dim": int(X_t.shape[1])}
            json.dump(state, OUT_A.open("w"), indent=1)
        print(f"{uid:<15} done ({len(state[uid])}/10)", flush=True)
    return state


def report_ablation(state):
    print(f"\n{'='*86}\nablation (GBDT, overall RMSE, mean of 5 seeds; "
          f"delta% against raw, negative = better)")
    hdr = ["raw"] + [f"+{f}" for f in FAMILIES] + [f"-{f}" for f in FAMILIES] + ["history"]
    print(f"{'unit':<15}" + "".join(f"{h:>11}" for h in hdr))
    agg = {h: [] for h in hdr}
    for uid, _, _ in FIT_UNITS:
        d = state[uid]
        base = d["raw"]["rmse_overall"]
        line = f"{uid:<15}"
        for h in hdr:
            v = d[h]["rmse_overall"]
            pct = (v - base) / base * 100
            agg[h].append(pct)
            line += f"{pct:>+11.1f}" if h != "raw" else f"{v:>11.5f}"
        print(line)
    print(f"{'median delta%':<15}" + "".join(
        f"{np.median(agg[h]):>+11.1f}" if h != "raw" else f"{'—':>11}" for h in hdr))
    print(f"{'units better':<15}" + "".join(
        f"{sum(1 for x in agg[h] if x < 0):>11}" if h != "raw" else f"{'—':>11}"
        for h in hdr))


def run_mechanism():
    b1 = [json.loads(l) for l in
          Path("results/battery/bexp1_author_protocol.jsonl").read_text().splitlines()]
    from collections import defaultdict
    pc = defaultdict(dict)
    for r in b1:
        if r["backbone"] != "tabpfn":
            continue
        for cell, v in r["per_cell"].items():
            pc[(r["unit"], cell, r["seed"])][r["view"]] = v
    gain = defaultdict(list)
    for (u, cell, seed), d in pc.items():
        if "raw" in d and "history" in d:
            gain[(u, cell)].append((d["history"] - d["raw"]) / d["raw"] * 100)

    rows = []
    for uid, ds, bk in FIT_UNITS:
        unit = load_fit_unit(ds, bk)
        for c in unit.test_cells:
            key = (uid, c.cell_id)
            if key not in gain or len(c.y) < 5:
                continue
            rate = abs(float(c.y[0]) - float(c.y[-1])) / len(c.y)  # loss per cycle
            rows.append({"unit": uid, "cell": c.cell_id,
                         "deg_rate": rate,
                         "gain_pct": float(np.mean(gain[key])),
                         "n_cycles": int(len(c.y))})
    x = [r["deg_rate"] for r in rows]
    y = [r["gain_pct"] for r in rows]
    rho = spearmanr(x, y)
    # within-unit correlation, to control for unit-level confounding
    within = []
    for uid, _, _ in FIT_UNITS:
        sub = [r for r in rows if r["unit"] == uid]
        if len(sub) >= 5:
            w = spearmanr([r["deg_rate"] for r in sub], [r["gain_pct"] for r in sub])
            within.append({"unit": uid, "n": len(sub),
                           "rho": float(w.statistic), "p": float(w.pvalue)})
    out = {"n_cells": len(rows),
           "pooled_rho": float(rho.statistic), "pooled_p": float(rho.pvalue),
           "within_unit": within, "cells": rows}
    json.dump(out, OUT_B.open("w"), indent=1)
    print(f"\n{'='*86}\nmechanism: per-cell history gain vs degradation rate "
          f"(n={len(rows)} cells)")
    print(f"  pooled Spearman rho={rho.statistic:+.3f} p={rho.pvalue:.2e} "
          f"(expected negative: faster fade, larger gain)")
    for w in within:
        print(f"  within {w['unit']:<15} n={w['n']:<3} rho={w['rho']:+.3f} "
              f"p={w['p']:.3f}")
    return out


if __name__ == "__main__":
    st = run_ablation()
    report_ablation(st)
    run_mechanism()
