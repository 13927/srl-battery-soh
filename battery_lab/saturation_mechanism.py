"""Why does the context saturate? Redundancy measures plus a sampling contrast.

Two candidate explanations for "adding rows beyond 8192 does nothing":
  H1 redundancy: consecutive cycles are so similar that new rows carry no new
     information
     measures: effective rank (feature spectrum), nearest-neighbour distance
     ratio, effective sample count after de-duplication
     falsifiable: if redundancy does not track the saturation point, H1 fails
  H2 distant rows hurt: far-away or heterogeneous cycles are stale information
     contrast: at the same 8192 budget, {random, recent (latest cycles per
     cell), early (earliest cycles)}
     falsifiable: if the three do not differ, H2 fails

Normalisation is strictly source-only throughout.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from battery_lab.data_adapters import FitUnit, normalize_cells
from battery_lab.temporal_features import build_view


def _stack(cells, Xn, view="raw", with_cycle_idx=False):
    xs, ys, cids, cyc = [], [], [], []
    for c in cells:
        xs.append(build_view(Xn[c.cell_id], view, cycle_mode="scaled200"))
        ys.append(c.y)
        cids.extend([c.cell_id] * len(c.y))
        cyc.extend(range(len(c.y)))
    out = [np.vstack(xs), np.concatenate(ys), np.array(cids)]
    if with_cycle_idx:
        out.append(np.array(cyc))
    return tuple(out)


# ---------------------------------------------------------------------------
# H1: redundancy measures
# ---------------------------------------------------------------------------

def redundancy_metrics(unit: FitUnit, seed: int = 0,
                       max_n: int = 8000) -> Dict[str, float]:
    """Quantify how redundant the training pool is.

    effective_rank       exponential of the spectral entropy (a more
                         concentrated spectrum means more redundancy and a lower
                         effective rank)
    neighbor_dist_ratio  nearest-neighbour distance over the global mean distance
                         (smaller = points more crowded = more redundant)
    dedup_ratio_1pct     fraction left after greedy de-duplication with a radius
                         at the 1st percentile of distances (smaller = more
                         redundant)
    rows_per_cell        mean cycles per cell (the direct source of redundancy)
    """
    cells = unit.train_cells
    Xn = normalize_cells(cells, "source", source_cells=cells)
    X, y, cids = _stack(cells, Xn, "raw")
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        idx = rng.choice(len(X), max_n, replace=False)
        X, cids = X[idx], cids[idx]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)

    # effective rank: exp(spectral entropy)
    s = np.linalg.svd(Xs, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 0]
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))

    # distance statistics
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(Xs)
    d1 = nn.kneighbors(Xs)[0][:, 1]          # nearest-neighbour distance (self excluded)
    sub = rng.choice(len(Xs), min(1500, len(Xs)), replace=False)
    D = np.sqrt(((Xs[sub][:, None, :] - Xs[sub][None, :, :]) ** 2).sum(-1))
    global_mean = float(D[np.triu_indices_from(D, 1)].mean())
    q1 = float(np.percentile(D[np.triu_indices_from(D, 1)], 1))

    # greedy de-duplication: anything within radius q1 counts as a duplicate
    keep = []
    order = rng.permutation(len(Xs))
    kept_pts = []
    for i in order:
        if not kept_pts:
            keep.append(i); kept_pts.append(Xs[i]); continue
        dmin = np.min(np.sqrt(((np.array(kept_pts) - Xs[i]) ** 2).sum(1)))
        if dmin > q1:
            keep.append(i); kept_pts.append(Xs[i])
        if len(kept_pts) > 3000:            # cost ceiling
            break

    return {
        "unit": unit.unit_id,
        "n_rows": int(len(X)),
        "effective_rank": eff_rank,
        "dim": int(X.shape[1]),
        "eff_rank_ratio": eff_rank / X.shape[1],
        "neighbor_dist_ratio": float(d1.mean() / global_mean),
        "dedup_ratio_1pct": float(len(keep) / len(Xs)),
        "rows_per_cell": float(len(X) / max(len(set(cids)), 1)),
    }


# ---------------------------------------------------------------------------
# H2: sampling contrast at a fixed budget
# ---------------------------------------------------------------------------

def sample_context(X, y, cyc, cids, mode: str, cap: int, seed: int):
    """Three samplers at the same budget: random / recent (latest cycles per
    cell) / early (earliest cycles per cell)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n <= cap:
        return X, y
    if mode == "random":
        idx = rng.choice(n, cap, replace=False)
    elif mode in ("recent", "early"):
        # sort each cell by cycle index and split the budget evenly
        cells = np.unique(cids)
        per = max(1, cap // len(cells))
        parts = []
        for c in cells:
            m = np.flatnonzero(cids == c)
            order = m[np.argsort(cyc[m])]
            parts.append(order[-per:] if mode == "recent" else order[:per])
        idx = np.concatenate(parts)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        elif len(idx) < cap:                      # top up
            rest = np.setdiff1d(np.arange(n), idx)
            idx = np.concatenate([idx, rng.choice(rest, cap - len(idx),
                                                  replace=False)])
    else:
        raise ValueError(mode)
    return X[idx], y[idx]


def sampling_comparison(unit: FitUnit, view: str, seed: int, cap: int = 8192,
                        modes=("random", "recent", "early")) -> List[Dict[str, Any]]:
    """Compare the samplers at a fixed budget (strict leave-one-cell-out, official
    splits)."""
    from residual_lab.models import make_model

    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "source", source_cells=tr)
    X_tr, y_tr, cid_tr, cyc_tr = _stack(tr, Xn, view, with_cycle_idx=True)
    X_te, y_te, cid_te = _stack(te, Xn, view)

    out = []
    for mode in modes:
        Xs, ys = sample_context(X_tr, y_tr, cyc_tr, cid_tr, mode, cap, seed)
        m = make_model("tabpfn_reg", seed=seed)
        m.fit(Xs, ys)
        pred = m.predict(X_te)
        per_cell = {c: float(np.sqrt(np.mean((y_te[cid_te == c]
                                             - pred[cid_te == c]) ** 2)))
                    for c in np.unique(cid_te)}
        out.append({"unit": unit.unit_id, "view": view, "mode": mode,
                    "cap": cap, "seed": seed, "n_ctx": int(len(ys)),
                    "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
                    "rmse_overall": float(np.sqrt(np.mean((y_te - pred) ** 2)))})
    return out
