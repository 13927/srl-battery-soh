"""Source-side configuration selection: nested leave-one-cell-out estimates plus
a selection rule.

Lineage: Caruana et al. 2004 (ensemble selection), with the validation set
replaced by nested leave-one-cell-out on the source side. Only source cells are
used throughout; target cells enter no estimate and no decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from battery_lab.data_adapters import FitUnit
from battery_lab.protocols import evaluate_unit

SAMPLING_SEED = 20260731
MAX_TABPFN_CELLS = 12       # cap on source cells for nested TabPFN runs (cost)
CONFIGS: List[Tuple[str, str]] = [
    ("tabpfn", "raw"), ("tabpfn", "history"),
    ("gbdt", "raw"), ("gbdt", "history"),
]


def subsample_cells(cells, k: int, seed: int = SAMPLING_SEED):
    """Subsample source cells with a fixed seed (reproducible, same seed
    convention as the view-selection rule)."""
    if len(cells) <= k:
        return list(cells)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(cells), k, replace=False))
    return [cells[i] for i in idx]


def loco_estimate(unit: FitUnit, backbone: str, view: str,
                  seeds, max_cells: Optional[int] = None) -> Dict[str, Any]:
    """Nested leave-one-cell-out over source cells: each source cell in turn
    becomes a pseudo target, the remaining source cells train the model.

    Returns the RMSE per (pseudo-target cell, seed) plus aggregates. It never
    touches unit.test_cells.
    """
    cells = list(unit.train_cells)
    if max_cells is not None:
        cells = subsample_cells(cells, max_cells)
    per = []                                   # (cell_id, seed, rmse)
    for i, tgt in enumerate(cells):
        srcs = [c for j, c in enumerate(cells) if j != i]
        for seed in seeds:
            r = evaluate_unit(unit, view, seed, backbone=backbone,
                              normalize="author_minmax",
                              train_cells=srcs, test_cells=[tgt])
            per.append({"cell": tgt.cell_id, "seed": seed,
                        "rmse": r["rmse_overall"]})
    vals = np.array([p["rmse"] for p in per])
    # aggregate per cell first, then take the macro mean and its SE (the cell is
    # the unit of independence)
    cell_means = {}
    for p in per:
        cell_means.setdefault(p["cell"], []).append(p["rmse"])
    cm = np.array([np.mean(v) for v in cell_means.values()])
    return {
        "backbone": backbone, "view": view,
        "n_cells": len(cells), "n_evals": len(per),
        "est_macro": float(cm.mean()),
        "se": float(cm.std(ddof=1) / np.sqrt(len(cm))) if len(cm) > 1 else 0.0,
        "per": per,
        "cells_used": [c.cell_id for c in cells],
    }


def select_config(estimates: Dict[str, Dict[str, Any]],
                  default: str = "gbdt-history") -> Dict[str, Any]:
    """Pre-specified selection rule: the smallest estimate wins; if its gap to the
    runner-up is below the pooled SE, fall back to the default configuration."""
    items = sorted(estimates.items(), key=lambda kv: kv[1]["est_macro"])
    (k1, e1), (k2, e2) = items[0], items[1]
    pooled_se = float(np.hypot(e1["se"], e2["se"]))
    if (e2["est_macro"] - e1["est_macro"]) < pooled_se:
        return {"selected": default, "reason": "tie->default",
                "gap": e2["est_macro"] - e1["est_macro"], "pooled_se": pooled_se,
                "ranking": [k for k, _ in items]}
    return {"selected": k1, "reason": "clear-min",
            "gap": e2["est_macro"] - e1["est_macro"], "pooled_se": pooled_se,
            "ranking": [k for k, _ in items]}
