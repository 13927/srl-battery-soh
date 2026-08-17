"""bexp18: LTT gate calibration and a retrospective comparison of three policies.

Calibration: leave-one-unit-out -- lambda* for a target unit is calibrated on the
cells of the other 10 units only (no new inference; the deployment story is to
calibrate on a public labelled library and apply it to one's own cells. Honest
note: exchangeability across units does not hold strictly, so the guarantee is
heuristic and the strict version would need nested leave-one-cell-out inside the
source).

Two gate directions:
  frozen  (the pre-registered direction): apply iso when v_pred <= lambda -- the
          frozen criteria G1-G3 are assessed on this
  revised (data-driven, post hoc): apply iso when v_pred >= lambda'
          basis = the phase-2 finding that v_pred correlates strongly and
          negatively with delta_iso (rho=-0.48, p=4.6e-6)
          -> v_pred is really a variance signal: the noisier the prediction, the
          more the monotone projection reduces variance.
          the revised direction may only be claimed after frozen verification on
          the unexposed library.

Three policies: never / always / gated, evaluated per cell, summarised per unit.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from battery_lab.conflict_gating import ltt_calibrate

DATA = Path("results/battery/bexp17_cell_level.jsonl")
OUT = Path("results/battery/bexp18_gating.json")
NOISE = 11.3


def load_cells():
    from collections import defaultdict
    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["unit"], r["cell_id"])].append(r)
    cells = []
    for (u, c), rs in agg.items():
        cells.append({
            "unit": u, "cell": c,
            "v_pred": float(np.mean([x["v_pred"] for x in rs])),
            "rmse_none": float(np.mean([x["rmse_none"] for x in rs])),
            "rmse_iso": float(np.mean([x["rmse_iso"] for x in rs])),
            "harm": bool(np.mean([x["delta_iso"] for x in rs]) > 0),
        })
    return cells


def evaluate(cells, decisions):
    """decisions: {(unit,cell): bool, whether iso is applied}. Returns per-unit
    macro RMSE."""
    units = sorted({c["unit"] for c in cells})
    out = {}
    for u in units:
        sub = [c for c in cells if c["unit"] == u]
        vals = [c["rmse_iso"] if decisions[(u, c["cell"])] else c["rmse_none"]
                for c in sub]
        out[u] = float(np.mean(vals))
    return out


def main():
    cells = load_cells()
    units = sorted({c["unit"] for c in cells})

    never = evaluate(cells, {(c["unit"], c["cell"]): False for c in cells})
    always = evaluate(cells, {(c["unit"], c["cell"]): True for c in cells})

    results = {"never": never, "always": always}
    for direction in ("frozen_le", "revised_ge"):
        dec, lam_by_unit = {}, {}
        for u in units:
            cal = [c for c in cells if c["unit"] != u]      # leave-one-unit-out
            if direction == "frozen_le":
                r = ltt_calibrate(np.array([c["v_pred"] for c in cal]),
                                  np.array([c["harm"] for c in cal]))
                lam = r["lam_star"]
                for c in [x for x in cells if x["unit"] == u]:
                    dec[(u, c["cell"])] = c["v_pred"] <= lam
            else:   # revised: apply when v_pred >= lambda'; equivalently LTT on -v_pred
                r = ltt_calibrate(-np.array([c["v_pred"] for c in cal]),
                                  np.array([c["harm"] for c in cal]))
                lam = -r["lam_star"]
                for c in [x for x in cells if x["unit"] == u]:
                    dec[(u, c["cell"])] = c["v_pred"] >= lam
            lam_by_unit[u] = float(lam)
        results[direction] = evaluate(cells, dec)
        results[f"{direction}_lambda"] = lam_by_unit
        results[f"{direction}_apply_rate"] = float(np.mean(list(dec.values())))

    # verdict
    print(f"{'unit':<16}{'never':>10}{'always':>10}{'frozen':>10}{'revised':>10}")
    g1f = g1r = True
    gains_f, gains_r = [], []
    for u in units:
        n, a = never[u], always[u]
        f, v = results["frozen_le"][u], results["revised_ge"][u]
        print(f"{u:<16}{n:>10.6f}{a:>10.6f}{f:>10.6f}{v:>10.6f}")
        if (f - n) / n * 100 > NOISE:
            g1f = False
        if (v - n) / n * 100 > NOISE:
            g1r = False
        if a < n:                     # units where always helps
            gains_f.append((n - f) / max(n - a, 1e-12))
            gains_r.append((n - v) / max(n - a, 1e-12))

    def wins(d):
        return sum(1 for u in units if d[u] <= min(never[u], always[u]) + 1e-12)

    for tag, res, g1, gains in [("frozen (pre-registered: v<=lambda)", results["frozen_le"], g1f, gains_f),
                                ("revised (post hoc: v>=lambda)", results["revised_ge"], g1r, gains_r)]:
        cap = float(np.median(gains)) if gains else float("nan")
        print(f"\n{tag}: apply rate={results['frozen_le_apply_rate' if 'frozen' in tag else 'revised_ge_apply_rate']:.0%}")
        print(f"  G1 do-no-harm (every unit vs never < +{NOISE}%): "
              + ("yes" if g1 else "no"))
        print(f"  G2 median gain captured (criterion >=60%): {cap:.0%}" if np.isfinite(cap) else "  G2 no unit where always helps")
        print(f"  G3 units where gated is no worse than either: {wins(res)}/11")

    json.dump(results, OUT.open("w"), indent=1)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
