"""View selection by nested leave-one-cell-out inside the source cells (official
test cells are never touched).

For one evaluation unit:
  1. take the official training cells S (sampled with a pre-specified seed when
     |S| > max_source_cells)
  2. each s in S becomes a pseudo target in turn: the context is the data of
     S minus {s} (<=2048 rows) and the whole trajectory of s is predicted
  3. both views give a per-pseudo-target RMSE -> source-side macro RMSE
  4. decision: cell-level paired bootstrap 95% CI
       CI entirely <0 -> history; entirely >0 -> raw; crossing zero -> abstain
       (average the two views' predictions)
  5. majority vote over 5 seeds; a tie gives abstain

The per-seed, per-cell differences are also kept, for the pre-registration
record and the forest plot of the paper.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from battery_lab.data_adapters import FitUnit
from battery_lab.protocols import evaluate_unit

MAX_SOURCE_CELLS = 30      # pre-specified: cap on source cells (CPU budget)
SAMPLING_SEED = 20260731   # pre-specified: sampling seed (independent of the
                           # evaluation seeds)
N_BOOT = 2000


def _select_source_cells(unit: FitUnit):
    cells = unit.train_cells
    if len(cells) <= MAX_SOURCE_CELLS:
        return cells
    rng = np.random.default_rng(SAMPLING_SEED)
    idx = rng.choice(len(cells), MAX_SOURCE_CELLS, replace=False)
    return [cells[i] for i in sorted(idx)]


def source_internal_loco(
    unit: FitUnit, seed: int, backbone: str = "tabpfn",
    normalize: str = "author_minmax",
) -> Dict[str, Any]:
    """One seed: nested leave-one-cell-out inside the source cells, returning the
    per-pseudo-target RMSE of both views."""
    cells = _select_source_cells(unit)
    per_cell = {"raw": {}, "history": {}}
    for i, target in enumerate(cells):
        sources = [c for j, c in enumerate(cells) if j != i]
        for view in ("raw", "history"):
            r = evaluate_unit(
                unit, view, seed, backbone=backbone, normalize=normalize,
                train_cells=sources, test_cells=[target])
            per_cell[view][target.cell_id] = r["rmse_cell_macro"]
    return per_cell


def decide_from_per_cell(per_cell: Dict[str, Dict[str, float]],
                         n_boot: int = N_BOOT, boot_seed: int = 0):
    """Cell-level paired bootstrap: diff = history - raw (negative favours
    history)."""
    ids = sorted(per_cell["raw"])
    diff = np.array([per_cell["history"][c] - per_cell["raw"][c] for c in ids])
    rng = np.random.default_rng(boot_seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if hi < 0:
        decision = "history"
    elif lo > 0:
        decision = "raw"
    else:
        decision = "abstain"
    return {"decision": decision, "mean_diff": float(diff.mean()),
            "ci": [float(lo), float(hi)], "n_cells": len(ids)}


def rule_D(unit: FitUnit, seeds=(0, 1, 2, 3, 4), backbone: str = "tabpfn",
           normalize: str = "author_minmax") -> Dict[str, Any]:
    """The full rule: source-side selection over several seeds plus a majority
    vote."""
    per_seed = []
    for seed in seeds:
        pc = source_internal_loco(unit, seed, backbone, normalize)
        d = decide_from_per_cell(pc, boot_seed=seed)
        d["seed"] = seed
        d["per_cell"] = pc
        per_seed.append(d)

    votes = [d["decision"] for d in per_seed]
    counts = {v: votes.count(v) for v in ("raw", "history", "abstain")}
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    final = winners[0] if len(winners) == 1 else "abstain"  # a tie gives abstain
    return {"unit": unit.unit_id, "backbone": backbone,
            "final_decision": final, "votes": votes, "per_seed": per_seed}
