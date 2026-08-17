"""bexp49: permutation inference for the mispairing arm on the external library
MATR-CLO -- one draw becomes a null distribution of 100.

Background: the external-library table reports point estimates over three
normalisations x three arms, but its mispaired arm rested on a single fixed draw
(the cyclic shift with MAP_SEED=20260731). That made it the one table in the paper
with a direction but no null distribution, while the main three-arm experiment
(bexp26/44, 11 units) already gives p=0.0099 from M=100 derangements. This script
brings the external library to the same standard.

Design (data cleaning, normalisation and assembly reuse the bexp41 functions, so
the pipeline is identical):
  normalisation: percell / source / sourcecyc (the three rows of bexp41)
  backbone:      deterministic GBDT (random_state=0, all training rows), as bexp41
  null:          M=100 independent derangements; draw m uses the cyclic shift with
                 seed MAP_SEED+m, so draw m=0 is exactly the bexp41 draw and can
                 be reconciled with that archive value by value
  statistic:     T = test RMSE (smaller is better)
            p = (1 + #{m: T_m <= T_self}) / (M + 1)

================ frozen criteria (committed before the run) ================
W1 (pipeline consistency): the raw/self/shuffled(m=0) values recomputed here match
   the corresponding entries of bexp41_matrclo_protocol.json to within 1e-9
   (3 values per normalisation, 9 in total)
   -> fails: the pipeline has drifted; stop interpreting and find the difference
      between bexp41 and this script first
W2 (main criterion): p <= 0.05 in the fully deployable cell (sourcecyc)
   -> holds: the self-pairing advantage on the external library is not the luck of
      one draw, and the table may carry a permutation p
   -> fails: report as measured, keep the point estimates and note that the null
      was not rejected
W3 (robustness): p <= 0.05 in all three normalisations
   -> if only percell fails: read it as the offline convention already supplying
      the self-comparison implicitly (the direction the carrier account predicts),
      write it that way, and do not treat it as a counter-example
============================================================

Run:
  ./.venv/bin/python experiments/bexp49_matrclo_permutation.py   # ~5 min
  ./.venv/bin/python experiments/bexp49_matrclo_permutation.py --report
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bexp41_matrclo_protocol import (  # noqa: E402  reuse the same pipeline
    MAP_SEED,
    assemble,
    load_cells,
    normalize,
    rmse,
)

REF = Path("results/battery/bexp41_matrclo_protocol.json")
OUT = Path("results/battery/bexp49_matrclo_permutation.json")
M = 100
MODES = ("percell", "source", "sourcecyc")


def derangement(ids, seed):
    """A derangement built as in bexp41: random permutation then cyclic shift (no
    fixed points). With seed=MAP_SEED this reproduces the bexp41 draw."""
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(ids))
    return {perm[i]: perm[(i + 1) % len(perm)] for i in range(len(perm))}


def fit_rmse(tr, te, Xn, arm, norm_mode, ref_map):
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_tr, y_tr = assemble(tr, Xn, arm, norm_mode, ref_map)
    X_te, y_te = assemble(te, Xn, arm, norm_mode, ref_map)
    m = HistGradientBoostingRegressor(random_state=0)
    m.fit(X_tr, y_tr)
    return rmse(y_te, m.predict(X_te))


def main():
    cells = load_cells()
    tr = [c for c in cells if not c["test"]]
    te = [c for c in cells if c["test"]]
    ids = [c["id"] for c in cells]
    state = json.load(OUT.open()) if OUT.exists() else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def save():
        json.dump(state, OUT.open("w"), indent=1)

    for norm_mode in MODES:
        Xn = normalize(cells, "percell" if norm_mode == "percell" else "source")
        for arm in ("raw", "self"):
            key = f"gbdt|{norm_mode}|{arm}"
            if key not in state:
                state[key] = fit_rmse(tr, te, Xn, arm, norm_mode,
                                      derangement(ids, MAP_SEED))
                save()
                print(f"{key:<26} rmse={state[key]:.5f}", flush=True)
        key_null = f"gbdt|{norm_mode}|null"
        vals = state.get(key_null, [])
        t0 = time.time()
        for m in range(len(vals), M):
            vals.append(fit_rmse(tr, te, Xn, "shuffled", norm_mode,
                                 derangement(ids, MAP_SEED + m)))
            state[key_null] = vals
            if (m + 1) % 20 == 0 or m + 1 == M:
                save()
                print(f"{key_null:<26} {m + 1}/{M} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    save()
    report(state)


def report(state=None):
    state = state or json.load(OUT.open())
    ref = json.load(REF.open()) if REF.exists() else {}
    print(f"\n{'=' * 78}\nMATR-CLO mispairing permutation test "
          f"(deterministic GBDT, M={M})")
    print(f"{'normalisation':<20}{'raw':>9}{'self':>9}{'null med':>10}"
          f"{'null range':>21}{'beats':>8}{'p':>8}")
    ps, drift = {}, []
    for nm in MODES:
        self_v = state.get(f"gbdt|{nm}|self")
        raw_v = state.get(f"gbdt|{nm}|raw")
        null = state.get(f"gbdt|{nm}|null") or []
        if self_v is None or len(null) < M:
            print(f"{nm:<20} incomplete (null={len(null)}/{M})")
            continue
        beaten = sum(1 for t in null if t <= self_v)
        p = (1 + beaten) / (M + 1)
        ps[nm] = p
        print(f"{nm:<14}{raw_v:>9.5f}{self_v:>9.5f}{np.median(null):>9.5f}"
              f"{f'[{min(null):.5f}, {max(null):.5f}]':>19}"
              f"{M - beaten:>4}/{M}{p:>8.4f}")
        # W1: reconcile with the bexp41 archive value by value (draw m=0 is the
        # bexp41 draw)
        for arm, val in (("raw", raw_v), ("self", self_v), ("shuffled", null[0])):
            got = ref.get(f"gbdt|{nm}|{arm}")
            if got is not None and abs(got - val) > 1e-9:
                drift.append(f"gbdt|{nm}|{arm}: bexp41={got:.9f} bexp49={val:.9f}")

    print(f"\n{'=' * 78}")
    print("W1 pipeline consistency (9 values against the bexp41 archive): "
          + ("all identical" if not drift else "drift: " + "; ".join(drift)))
    if "sourcecyc" in ps:
        print(f"W2 main criterion (sourcecyc p<=0.05): "
              f"{'✓' if ps['sourcecyc'] <= 0.05 else '✗'} p={ps['sourcecyc']:.4f}")
    if len(ps) == len(MODES):
        n_ok = sum(1 for p in ps.values() if p <= 0.05)
        print(f"W3 robustness (p<=0.05 in 3/3 normalisations): "
              f"{'✓' if n_ok == 3 else f'✗ {n_ok}/3'}"
              + ("" if n_ok == 3 or ps.get("percell", 0) > 0.05
                 else "  note: the failing cell is not percell, which "
                      "contradicts the carrier account and needs checking"))
    print("=" * 78)


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
