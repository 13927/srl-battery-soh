"""Context selection for battery rows: random / kNN retrieval / clustering, with
a sweep over k and an adaptive rule driven by distribution descriptors.

Three questions are addressed:
  1. retrieval strategy -- per-test-point kNN retrieval plus clustering, over a
     range of k
  2. descriptors of the data -- group_shift_auc / knn_locality /
     degradation_rate
  3. adapting k -- predict the best k per unit from those descriptors

Strict mode throughout (normalisation and cycle column are source-only, matching
the frozen protocol). The context budget is the number of retrieved rows (for
kNN: k_per_test after de-duplication; everything is capped by CONTEXT_CAP).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from battery_lab.data_adapters import CellData, FitUnit, normalize_cells
from battery_lab.temporal_features import build_view

CTX_HARD_CAP = 8192   # hard cap on rows in one context (above the 2048 default)


def _assemble(cells, Xn, view):
    xs, ys, ids = [], [], []
    for c in cells:
        xs.append(build_view(Xn[c.cell_id], view, cycle_mode="scaled200"))
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def retrieval_predict(
    unit: FitUnit, view: str, seed: int, strategy: str = "knn",
    k_per_test: int = 50, n_clusters: int = 10,
    model_name: str = "tabpfn_reg",
) -> Dict[str, Any]:
    """Build a context for the test rows with the given strategy and predict,
    under strict leave-one-cell-out.

    strategy:
      random   sample min(CTX_HARD_CAP, pool) rows at random (baseline)
      knn      cluster the test rows into n_clusters, then take the union of the
               k_per_test neighbours of every member
      cluster  take the CTX_HARD_CAP neighbours of each cluster centroid
               (MixturePFN style, for contrast)
    """
    from residual_lab.models import make_model

    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "source", source_cells=tr)
    X_pool, y_pool, _ = _assemble(tr, Xn, view)
    X_te, y_te, te_ids = _assemble(te, Xn, view)
    # distance space: standardise the base features (first 17 columns) so the
    # high-dimensional history block cannot dominate the distances
    base_dim = 17
    mu, sd = X_pool[:, :base_dim].mean(0), X_pool[:, :base_dim].std(0) + 1e-9
    Ps = (X_pool[:, :base_dim] - mu) / sd
    Ts = (X_te[:, :base_dim] - mu) / sd
    rng = np.random.default_rng(seed)

    pred = np.zeros(len(X_te))

    if strategy == "random":
        cap = min(CTX_HARD_CAP, len(X_pool))
        idx = rng.choice(len(X_pool), cap, replace=False)
        m = make_model(model_name, seed=seed)
        m.fit(X_pool[idx], y_pool[idx])
        pred = m.predict(X_te)

    elif strategy in ("knn", "cluster"):
        from sklearn.cluster import KMeans
        nc = min(n_clusters, len(X_te))
        labels = KMeans(n_clusters=nc, random_state=seed,
                        n_init=3).fit_predict(Ts)
        for c in range(nc):
            mask = labels == c
            if not mask.any():
                continue
            if strategy == "knn":
                chosen = []
                for m_s in Ts[mask]:
                    d = ((Ps - m_s) ** 2).sum(1)
                    chosen.append(np.argpartition(d, min(k_per_test, len(d)-1))[:k_per_test])
                idx = np.unique(np.concatenate(chosen))
            else:  # cluster: neighbours of the centroid
                centroid = Ts[mask].mean(0)
                d = ((Ps - centroid) ** 2).sum(1)
                idx = np.argpartition(d, min(CTX_HARD_CAP, len(d)-1))[:CTX_HARD_CAP]
            if len(idx) > CTX_HARD_CAP:
                idx = rng.choice(idx, CTX_HARD_CAP, replace=False)
            mdl = make_model(model_name, seed=seed)
            mdl.fit(X_pool[idx], y_pool[idx])
            pred[mask] = mdl.predict(X_te[mask])
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    per_cell = {cid: float(np.sqrt(np.mean((y_te[te_ids == cid]
                                           - pred[te_ids == cid]) ** 2)))
                for cid in np.unique(te_ids)}
    return {"unit": unit.unit_id, "view": view, "strategy": strategy,
            "k_per_test": k_per_test, "n_clusters": n_clusters, "seed": seed,
            "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
            "rmse_overall": float(np.sqrt(np.mean((y_te - pred) ** 2))),
            "ctx_rows_est": int(min(CTX_HARD_CAP, len(X_pool)))}


def distribution_indicators(unit: FitUnit, seed: int = 0) -> Dict[str, float]:
    """Describe the data distribution, for adapting k. Uses source cells and the
    target features only, never target labels."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsRegressor

    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "source", source_cells=tr)
    Xp, yp, _ = _assemble(tr, Xn, "raw")
    Xt, yt, _ = _assemble(te, Xn, "raw")
    base = 17
    Xp, Xt = Xp[:, :base], Xt[:, :base]

    # descriptor 1: source-target shift (group classifier AUC; higher = more shift)
    Xg = np.vstack([Xp, Xt])
    yg = np.r_[np.zeros(len(Xp)), np.ones(len(Xt))]
    sub = np.random.default_rng(seed).choice(len(Xg), min(4000, len(Xg)), replace=False)
    try:
        auc = float(np.mean(cross_val_score(
            HistGradientBoostingClassifier(max_iter=80, random_state=seed),
            Xg[sub], yg[sub], cv=3, scoring="roc_auc")))
    except ValueError:
        auc = 0.5

    # descriptor 2: locality (kNN regression R2 within the source; high means a
    # strong local structure, where kNN retrieval ought to help)
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-9
    knn_r2 = float(np.mean(cross_val_score(
        KNeighborsRegressor(n_neighbors=10), (Xp - mu) / sd, yp, cv=3, scoring="r2")))

    # descriptor 3: degradation rate (mean of end-to-end SOH change over cycles
    # across source cells)
    rates = []
    for c in tr:
        if len(c.y) > 5:
            rates.append(abs(c.y[0] - c.y[-1]) / len(c.y))
    deg_rate = float(np.mean(rates)) if rates else 0.0

    return {"group_shift_auc": auc, "knn_locality_r2": knn_r2,
            "degradation_rate": deg_rate, "pool_rows": int(len(Xp))}


def adaptive_k(ind: Dict[str, float]) -> int:
    """Adaptive rule: strong locality gives a small k (more focused), weak
    locality a large k (more global).

    high knn_locality_r2 -> the neighbourhood is informative -> small k
    strong shift (high auc) -> the target sits far from the source, so more
    neighbours are needed -> large k
    """
    loc = ind["knn_locality_r2"]
    auc = ind["group_shift_auc"]
    if loc > 0.8:
        k = 30
    elif loc > 0.5:
        k = 60
    else:
        k = 120
    if auc > 0.85:          # large shift, widen the neighbourhood
        k = int(k * 1.5)
    return int(np.clip(k, 20, 300))
