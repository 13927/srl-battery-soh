"""bexp28: a robust self-anchor, and the leaner lift.

Method: from_first_K = X_t - running_median(X_1..min(t+1,K))
      -- for t<K the anchor uses cycles <=t only (which keeps it history-only);
      from t>=K the anchor is fixed at the median of the first K cycles. K=1
      reduces exactly to the incumbent from_first.

================ frozen criteria (committed before the run) ================
The pre-registered main setting is K=5 (K in {3,10} are sensitivity checks only,
so that K cannot be picked after the fact).
A1 (main):    with GBDT over 11 units, K=5 is no worse than K=1 (delta<=0) in >= 7
              of 11 units
A2 (repair):  at least one of HUST or TJU-3 flips from "the self arm hurts" to
              delta% <= 0 against raw
A3 (across models): spot-check TabPFN on TJU-1/HUST/MIT x 3 seeds; K=5 is no worse
              than K=1 in >= 2 of 3
L1 (leaner):  lean+ = raw + ffK* + roll5 (49d) differs from the 81d history stack by
              a unit-level median of less than 11.3 per cent
Rule for choosing K* and the construction (fixed in advance): K* is the value among
  {ff1,ff3,ff5,ff10} with the best unit-level median RMSE; the construction is the
  best median among {lean(33d), lean+(49d), history(81d)}, ties going to the
  simpler one.
=======================================================

The raw / +from_first (=ff1) / history results of bexp24 are reused (same protocol,
same seeds).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit, normalize_cells
from battery_lab.protocols import (CONTEXT_CAP, fit_predict_gbdt,
                                   fit_predict_tabpfn, subsample_context)
from battery_lab.temporal_features import cycle_author_minmax

OUT = Path("results/battery/bexp28_rsr.json")
B24 = Path("results/battery/bexp24_ablation.json")
SEEDS = [0, 1, 2, 3, 4]
TAB_UNITS = ["TJU-1", "HUST", "MIT"]
TAB_SEEDS = [0, 1, 2]
NOISE = 11.3


def from_first_k(Z, K):
    """History-only robust anchor: the anchor of row t is
    median(Z[0..min(t, K-1)])."""
    n = Z.shape[0]
    out = np.empty_like(Z)
    anchor_full = np.median(Z[:K], axis=0)
    for t in range(n):
        if t + 1 >= K:
            out[t] = Z[t] - anchor_full
        else:
            out[t] = Z[t] - np.median(Z[:t + 1], axis=0)
    return out


def roll5_block(Z):
    csum = np.cumsum(Z, axis=0)
    out = np.empty_like(Z)
    for t in range(Z.shape[0]):
        lo = max(0, t - 4)
        total = csum[t] - (csum[lo - 1] if lo > 0 else 0)
        out[t] = Z[t] - total / (t - lo + 1)
    return out


def assemble(cells, Xn, cond):
    xs, ys = [], []
    for c in cells:
        Z = Xn[c.cell_id]
        cols = [Z, cycle_author_minmax(len(Z))]
        if cond.startswith("ff"):
            cols.append(from_first_k(Z, int(cond[2:])))
        elif cond.startswith("leanplus"):
            K = int(cond.split("_K")[1])
            cols += [from_first_k(Z, K), roll5_block(Z)]
        xs.append(np.hstack(cols))
        ys.append(c.y)
    return np.vstack(xs), np.concatenate(ys)


def eval_cond(unit, Xn, cond, backbone="gbdt", seeds=SEEDS):
    fit = fit_predict_gbdt if backbone == "gbdt" else fit_predict_tabpfn
    X_tr, y_tr = assemble(unit.train_cells, Xn, cond)
    X_te, y_te = assemble(unit.test_cells, Xn, cond)
    vals = []
    for seed in seeds:
        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
        pred = fit(Xc, yc, X_te, seed)
        vals.append(float(np.sqrt(np.mean((y_te - pred) ** 2))))
    return float(np.mean(vals))


def main():
    state = json.load(OUT.open()) if OUT.exists() else {}
    b24 = json.load(B24.open())
    umap = {u: (d, b) for u, d, b in FIT_UNITS}
    units = [u for u, _, _ in FIT_UNITS]

    # stage 1: the K grid (ff1 comes from bexp24)
    for uid in units:
        st = state.setdefault(uid, {})
        st.setdefault("raw", b24[uid]["raw"]["rmse_overall"])
        st.setdefault("ff1", b24[uid]["+from_first"]["rmse_overall"])
        st.setdefault("history", b24[uid]["history"]["rmse_overall"])
        todo = [c for c in ("ff3", "ff5", "ff10") if c not in st]
        if not todo:
            continue
        unit = load_fit_unit(*umap[uid])
        Xn = normalize_cells(unit.train_cells + unit.test_cells, "author_minmax")
        t0 = time.time()
        for c in todo:
            st[c] = eval_cond(unit, Xn, c)
            json.dump(state, OUT.open("w"), indent=1)
        print(f"{uid:<15} ff1={st['ff1']:.5f} ff3={st['ff3']:.5f} "
              f"ff5={st['ff5']:.5f} ff10={st['ff10']:.5f} [{time.time()-t0:.0f}s]",
              flush=True)

    # K* by the pre-registered rule: best unit-level median
    med = {K: float(np.median([(state[u][f"ff{K}"] - state[u]["raw"])
                               / state[u]["raw"] for u in units]))
           for K in (1, 3, 5, 10)}
    k_star = min(med, key=med.get)
    state["_meta"] = {"median_delta_by_K": med, "K_star": k_star}
    json.dump(state, OUT.open("w"), indent=1)

    # stage 2: lean+ (49d, using K*)
    for uid in units:
        st = state[uid]
        cond = f"leanplus_K{k_star}"
        if cond in st:
            continue
        unit = load_fit_unit(*umap[uid])
        Xn = normalize_cells(unit.train_cells + unit.test_cells, "author_minmax")
        st[cond] = eval_cond(unit, Xn, cond)
        json.dump(state, OUT.open("w"), indent=1)
        print(f"{uid:<15} leanplus_K{k_star}={st[cond]:.5f}", flush=True)

    # stage 3: the TabPFN spot-check
    for uid in TAB_UNITS:
        st = state[uid]
        for c in ("ff1", "ff5"):
            key = f"tab_{c}"
            if key in st:
                continue
            unit = load_fit_unit(*umap[uid])
            Xn = normalize_cells(unit.train_cells + unit.test_cells,
                                 "author_minmax")
            st[key] = eval_cond(unit, Xn, c, backbone="tabpfn",
                                seeds=TAB_SEEDS)
            json.dump(state, OUT.open("w"), indent=1)
            print(f"{uid:<15} {key}={st[key]:.5f}", flush=True)
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    units = [u for u, _, _ in FIT_UNITS]
    k_star = state["_meta"]["K_star"]
    print(f"\n{'='*84}\nrobust-anchor verdict "
          f"(delta% against raw, negative = better; K*={k_star})")
    print(f"{'unit':<16}{'ff1':>8}{'ff3':>8}{'ff5':>8}{'ff10':>8}"
          f"{'lean+':>8}{'history':>9}")
    a1 = 0
    for uid in units:
        st = state[uid]
        b = st["raw"]
        d = {c: (st[c] - b) / b * 100 for c in
             ("ff1", "ff3", "ff5", "ff10", f"leanplus_K{k_star}", "history")}
        if st["ff5"] <= st["ff1"] + 1e-12:
            a1 += 1
        print(f"{uid:<16}" + "".join(f"{d[c]:>+8.1f}" for c in
              ("ff1", "ff3", "ff5", "ff10")) +
              f"{d[f'leanplus_K{k_star}']:>+8.1f}{d['history']:>+9.1f}")
    print(f"\nA1 (K=5 no worse than K=1 in >=7/11): {a1}/11 -> "
          + ("holds" if a1 >= 7 else "falsified"))
    for uid in ("HUST", "TJU-3"):
        st = state[uid]
        d5 = (st["ff5"] - st["raw"]) / st["raw"] * 100
        d1 = (st["ff1"] - st["raw"]) / st["raw"] * 100
        print(f"A2 {uid}: ff1={d1:+.1f}% → ff5={d5:+.1f}% "
              f"{'flipped' if d5 <= 0 else 'not flipped'}")
    a3 = 0
    for uid in TAB_UNITS:
        st = state[uid]
        if "tab_ff5" in st and "tab_ff1" in st:
            ok = st["tab_ff5"] <= st["tab_ff1"] + 1e-12
            a3 += int(ok)
            print(f"A3 tabpfn {uid:<8} ff1={st['tab_ff1']:.5f} "
                  f"ff5={st['tab_ff5']:.5f} {'✓' if ok else '✗'}")
    print(f"A3: {a3}/{len(TAB_UNITS)} (criterion >=2)")
    lp = [((state[u][f"leanplus_K{k_star}"] - state[u]["history"])
           / state[u]["history"] * 100) for u in units]
    print(f"L1 lean+ vs history(81d): median {np.median(lp):+.1f}% "
          f"(criterion |median|<{NOISE}) -> "
          + ("holds" if abs(np.median(lp)) < NOISE else "falsified"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
