"""bexp23: frozen verification of the fixed ensemble recipe (ens-mean) on the
unexposed library.

Frozen predictions: Q1 the bias-diversity identity holds; Q2 ens versus its best
member < +11.1 per cent; Q3 ens beats both members; Q4 ens < 0.014548 (the bexp13
single-model history baseline).
Run once, 3 seeds. GBDT uses the same strict feature matrix as bexp5 (same folds,
same budget).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from battery_lab.protocols import fit_predict_gbdt
from bexp5_unexposed_frozen import (CONTEXT_CAP, assemble, load_unit_npz,
                                    prepare, subsample_context)

OUT = Path("results/battery/bexp23_scs_validation.json")
SEEDS = [0, 1, 2]
BASE_HIST = 0.014548     # frozen bexp13 baseline


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    from residual_lab.models import make_model

    unit = load_unit_npz()
    src, tgt = unit.train_cells, unit.test_cells
    normed, ind = prepare(unit, src, tgt)
    X_tr, y_tr, _ = assemble(src, normed, ind, "history")
    X_te, y_te, _ = assemble(tgt, normed, ind, "history")

    per_seed = []
    for seed in SEEDS:
        Xc, yc = subsample_context(X_tr, y_tr, CONTEXT_CAP, seed)
        m = make_model("tabpfn_reg", seed=seed)
        m.fit(Xc, yc)
        pt = m.predict(X_te)
        pg = fit_predict_gbdt(Xc, yc, X_te, seed)
        pe = (np.asarray(pt) + np.asarray(pg)) / 2
        per_seed.append({"seed": seed, "tab": rmse(pt, y_te),
                         "gbdt": rmse(pg, y_te), "ens": rmse(pe, y_te)})
        print(f"seed={seed} tab={per_seed[-1]['tab']:.6f} "
              f"gbdt={per_seed[-1]['gbdt']:.6f} ens={per_seed[-1]['ens']:.6f}",
              flush=True)

    tab = float(np.mean([s["tab"] for s in per_seed]))
    gbd = float(np.mean([s["gbdt"] for s in per_seed]))
    ens = float(np.mean([s["ens"] for s in per_seed]))
    best = min(tab, gbd)
    verdict = {
        "tab": tab, "gbdt": gbd, "ens": ens,
        "Q1_kv": bool(ens <= (tab + gbd) / 2 + 1e-12),
        "Q2_vs_best_pct": (ens - best) / best * 100,
        "Q2_holds": bool((ens - best) / best * 100 < 11.1),
        "Q3_beats_both": bool(ens < tab and ens < gbd),
        "Q4_vs_base": bool(ens < BASE_HIST),
        "per_seed": per_seed,
    }
    json.dump(verdict, OUT.open("w"), indent=1)
    print(f"\ntab={tab:.6f} gbdt={gbd:.6f} ens={ens:.6f}")
    print("Q1 bias-diversity identity: "
          + ("holds" if verdict['Q1_kv'] else "BUG"))
    print(f"Q2 ens vs best member {verdict['Q2_vs_best_pct']:+.2f}% (<+11.1%): "
          + ("holds" if verdict['Q2_holds'] else "falsified"))
    print("Q3 ens beats both members: "
          + ("holds" if verdict['Q3_beats_both'] else "no"))
    print(f"Q4 ens < baseline {BASE_HIST}: "
          + ("holds" if verdict['Q4_vs_base'] else "falsified"))


if __name__ == "__main__":
    main()
