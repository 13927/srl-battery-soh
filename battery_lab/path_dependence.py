"""Path dependence: do cross-cycle features separate neighbours that look alike
as snapshots but differ in SOH?

The measure is neighbour label discordance: for every cycle i, take its k
nearest neighbours in a given feature space and average |SOH(neighbour) -
SOH(i)|. If looking alike implied equal SOH, the value would be small; strong
path dependence makes snapshot neighbours differ widely in SOH.

Reading:
  large discordance in the raw space   -> a single-cycle snapshot does not
                                          determine SOH (path dependence exists)
  discordance(history) < raw           -> the cross-cycle features pulled those
                                          contradictory neighbours apart
                                          (mechanistic evidence)

"Snapshot twins" are also counted: pairs whose feature distance is below a
quantile threshold while their SOH differs by more than a threshold, compared
between the raw and history spaces.

Normalisation is strictly source-only. The measurement runs over all cells of a
unit (train and test alike), because this analyses the data structure rather
than making a prediction.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from battery_lab.data_adapters import FitUnit, normalize_cells
from battery_lab.temporal_features import build_view


def _assemble(cells, Xn, view):
    xs, ys, cids = [], [], []
    for c in cells:
        xs.append(build_view(Xn[c.cell_id], view, cycle_mode="scaled200"))
        ys.append(c.y)
        cids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(cids)


def neighbor_discordance(
    X: np.ndarray, y: np.ndarray, cids: np.ndarray, k: int = 10,
    max_n: int = 5000, seed: int = 0, exclude_same_cell: bool = True,
) -> Dict[str, float]:
    """Neighbour label discordance plus the snapshot-twin count.

    exclude_same_cell: drop neighbours from the same cell. Without this, the
      cross-cycle features would trivially pull neighbouring cycles of one cell
      together and deflate the discordance, which would not be a fair
      comparison. This is the key fairness control.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        idx = rng.choice(len(X), max_n, replace=False)
        X, y, cids = X[idx], y[idx], cids[idx]

    # standardise to unit variance so that scale cannot dominate the distances
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd

    n_query = min(len(X), 1500)     # cost control: sample the query points
    qidx = rng.choice(len(X), n_query, replace=False)

    # ask for extra neighbours so that k remain after same-cell ones are dropped
    nn = NearestNeighbors(n_neighbors=min(k * 5 + 1, len(X))).fit(Xs)
    dist, ind = nn.kneighbors(Xs[qidx])

    discords, pair_dists, pair_soh_gaps = [], [], []
    for qi, (drow, irow) in enumerate(zip(dist, ind)):
        self_idx = qidx[qi]
        self_soh = y[self_idx]
        self_cell = cids[self_idx]
        picked = []
        for d, j in zip(drow, irow):
            if j == self_idx:
                continue
            if exclude_same_cell and cids[j] == self_cell:
                continue
            picked.append((d, j))
            if len(picked) >= k:
                break
        if not picked:
            continue
        soh_gaps = [abs(y[j] - self_soh) for _, j in picked]
        discords.append(np.mean(soh_gaps))
        # twins: feature distance and SOH gap of the nearest neighbour from
        # another cell
        pair_dists.append(picked[0][0])
        pair_soh_gaps.append(abs(y[picked[0][1]] - self_soh))

    discords = np.array(discords)
    pair_dists = np.array(pair_dists)
    pair_soh_gaps = np.array(pair_soh_gaps)

    # snapshot twins: feature distance in the closest 20 per cent, yet the SOH gap
    # exceeds 0.05 (5 per cent of SOH)
    close = pair_dists <= np.percentile(pair_dists, 20)
    twins = close & (pair_soh_gaps > 0.05)
    return {
        "mean_neighbor_soh_gap": float(discords.mean()),
        "median_neighbor_soh_gap": float(np.median(discords)),
        "twin_rate": float(twins.mean()),        # share of close-but-distant pairs
        "n_query": int(len(discords)),
    }


def analyze_unit(unit: FitUnit, k: int = 10, seed: int = 0) -> Dict[str, Any]:
    """Compare neighbour discordance between the raw and history spaces on one
    evaluation unit."""
    cells = unit.cells
    Xn = normalize_cells(cells, "source", source_cells=cells)
    out = {"unit": unit.unit_id, "n_cells": len(cells)}
    for view in ("raw", "history"):
        X, y, cids = _assemble(cells, Xn, view)
        out[view] = neighbor_discordance(X, y, cids, k=k, seed=seed)
    r, h = out["raw"], out["history"]
    out["gap_reduction_pct"] = (
        (h["mean_neighbor_soh_gap"] - r["mean_neighbor_soh_gap"])
        / r["mean_neighbor_soh_gap"] * 100)
    out["twin_reduction_pct"] = (
        (h["twin_rate"] - r["twin_rate"]) / (r["twin_rate"] + 1e-9) * 100)
    return out
