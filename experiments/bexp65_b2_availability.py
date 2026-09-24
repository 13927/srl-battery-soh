"""bexp65: F3-02 acceptance check -- "raw cycle 100" availability for B2 scoring rows.

Background: bexp63's scoring set is defined by "retained-record index >= 100" (0-based), while the
dQ(V) scalar uses the discharge curves of **raw cycles 10 and 100**, which is available only after
the 100th raw cycle has finished discharging. Checking "retained-record index >= 100" alone does
not establish that "raw cycle 100 has completed at every scoring row" -- this script verifies with
the manifest's explicit row mapping (align.raw_index_of_proc):

  1) the raw-cycle range corresponding to the scoring rows of each B2 scoring cell (a test cell
     with a scalar);
  2) min raw cycle <= 100 implies a risk that scoring rows precede the scalar's availability
     (information leakage) -> violation;
  3) identity/mapping-failure cells from manifest.release_gate should be excluded from B2 -- the
     checked list;
  4) cells with no raw curve data (no_raw_curve_data) and no scalar (no_scalar) are logged as
     measured.

================== criteria (frozen before the run) ==================
A1 per (unit, cell) reporting: scoring row count, mapped row count, raw cycle min/max;
A2 violation: any scoring row with raw cycle number <= 100 (the scoring start should be >= 101);
A3 violation: scoring-row mapping coverage < 100% (unmappable rows exist);
A4 output `results/battery/bexp65_b2_availability.json`; non-empty violations -> exit code 1.
==================================================================

Run:  ./.venv/bin/python experiments/bexp65_b2_availability.py [--report]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from battery_lab.data_adapters import FIT_UNITS, load_fit_unit

MANIFEST = ROOT / "results/battery/raw_data_manifest.json"
CURVE = ROOT / "results/battery/bexp62_curve_features.json"
OUT = ROOT / "results/battery/bexp65_b2_availability.json"
N0 = 100
READER = "strict_online"


def load_maps():
    man = json.load(open(MANIFEST))
    maps, excluded = {}, set()
    for cell, v in man["xjtu"].items():
        al = v.get("align")
        if al:
            maps[cell] = al
    for b, grp in man["mit"].items():
        for c in grp["cells"]:
            al = c.get("align")
            if al:
                maps[f"{b}_battery-{c['order_index']}"] = al
    gate = man.get("release_gate") or {}
    for key in ("identity_failures", "unverified"):
        for r in gate.get(key, []):
            excluded.add(r["cell"])
    return maps, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report and OUT.exists():
        out = json.load(open(OUT))
        return report(out)

    maps, excluded = load_maps()
    curve = json.load(open(CURVE))["cells"]
    rows = []
    for uid, ds, bk in FIT_UNITS:
        unit = load_fit_unit(ds, bk, reader=READER)
        for c in unit.test_cells:
            rec = {"unit": uid, "cell": c.cell_id}
            if c.cell_id in excluded:
                rec.update({"status": "excluded_identity"})
                rows.append(rec)
                continue
            al = maps.get(c.cell_id)
            if al is None:
                rec.update({"status": "no_raw_curve_data"})
                rows.append(rec)
                continue
            if curve.get(c.cell_id, {}).get("qdiff_var_log10") is None:
                rec.update({"status": "no_scalar"})
                rows.append(rec)
                continue
            pos = np.arange(len(c.y))
            sel = pos >= N0
            if sel.sum() == 0:
                rec.update({"status": "no_scored_rows"})
                rows.append(rec)
                continue
            ridx = np.asarray(c.record_index)
            raw = np.asarray(al["raw_index_of_proc"], dtype=int)
            csv_rows = ridx[sel]
            inb = (csv_rows >= 0) & (csv_rows < raw.size)
            rc = np.full(csv_rows.size, -1, dtype=int)
            rc[inb] = raw[csv_rows[inb]]
            n_map = int((rc >= 0).sum())
            viol = []
            if n_map < int(sel.sum()):
                viol.append(f"unmapped_rows={int(sel.sum()) - n_map}")
            if n_map and int(rc[rc >= 0].min()) <= N0:
                viol.append(f"min_raw_cycle={int(rc[rc >= 0].min())}<=100")
            rec.update({"n_scored_rows": int(sel.sum()), "n_mapped": n_map,
                        "raw_cycle_min": int(rc[rc >= 0].min()) if n_map else None,
                        "raw_cycle_max": int(rc[rc >= 0].max()) if n_map else None,
                        "violations": viol, "status": "violation" if viol else "ok"})
            rows.append(rec)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    viol = [r for r in rows if r.get("status") == "violation"]
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = {"_meta": {"script": "bexp65_b2_availability.py",
                     "note": "raw-cycle availability check for B2 scoring rows (min raw cycle should be >= 101)",
                     "n0": N0},
           "counts": counts, "cells": rows, "violations": viol}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[bexp65] cells={len(rows)} ok={n_ok} violations={len(viol)} "
          f"counts={counts}")
    for r in viol[:10]:
        print("  VIOLATION:", r)
    for r in rows:
        if r["status"] == "ok":
            print(f"  ok {r['unit']:10s} {r['cell']:24s} rows={r['n_scored_rows']} "
                  f"raw_cyc=[{r['raw_cycle_min']},{r['raw_cycle_max']}]")
    return 1 if viol else 0


def report(out):
    print(f"[bexp65] counts={out['counts']} violations={len(out['violations'])}")
    for r in out["violations"]:
        print("  VIOLATION:", r)
    return 1 if out["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
