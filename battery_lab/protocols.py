"""Reference-protocol evaluation: assemble train/test matrices, run the
TabPFN-2048 / MLP backbones, report both metrics.

Protocol details (matching the convention of the reported numbers):
  - Features: the 16 statistics after the reference authors' per-cell
    full-lifetime min-max; the history view expands them across cycles
    (within a cell, history-only)
  - Context: training rows subsampled at random to <=2048 (seed-controlled)
  - Metrics: overall = RMSE over all pooled test rows (the authors'
    sample-weighted convention)
             cell_macro = macro average of per-cell RMSE (the XJTU convention)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from battery_lab.data_adapters import CellData, FitUnit, normalize_cells
from battery_lab.temporal_features import build_view

CONTEXT_CAP = 2048  # frozen: a 10k context once exhausted the machine


def assemble_matrix(
    cells: List[CellData], Xnorm: Dict[str, np.ndarray], view: str,
    cycle_mode: str = "scaled200",
):
    """Stack the view feature matrix cell by cell; returns X, y, cell_ids (per row)."""
    xs, ys, ids = [], [], []
    for c in cells:
        Xv = build_view(Xnorm[c.cell_id], view, cycle_mode=cycle_mode)
        xs.append(Xv)
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def subsample_context(X, y, cap: int, seed: int):
    if len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), cap, replace=False)
    return X[idx], y[idx]


def fit_predict_tabpfn(X_tr, y_tr, X_te, seed: int):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from residual_lab.models import make_model

    m = make_model("tabpfn_reg", seed=seed)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def fit_predict_mlp(X_tr, y_tr, X_te, seed: int):
    """Feature-matched MLP baseline (sklearn, standardisation + two hidden layers)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                     early_stopping=True, random_state=seed),
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_te)

def fit_predict_gbdt(X_tr, y_tr, X_te, seed: int):
    """Feature-matched GBDT baseline (HistGradientBoostingRegressor, defaults).

    Using library defaults is deliberate: it keeps the treatment symmetric with
    TabPFN (n_estimators=2, untuned) and the MLP (default regularisation), so
    that no family is tuned.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


BACKBONES = {"tabpfn": fit_predict_tabpfn, "mlp": fit_predict_mlp,
             "gbdt": fit_predict_gbdt}


def evaluate_unit(
    unit: FitUnit, view: str, seed: int,
    backbone: str = "tabpfn",
    normalize: str = "author_minmax",
    context_cap: int = CONTEXT_CAP,
    train_cells: Optional[List[CellData]] = None,
    test_cells: Optional[List[CellData]] = None,
) -> Dict[str, Any]:
    """Evaluate one (view, backbone, seed) combination on one evaluation unit.

    train_cells/test_cells may override the official split (reused by the
    nested leave-one-cell-out selection inside the source cells).
    """
    tr_cells = train_cells if train_cells is not None else unit.train_cells
    te_cells = test_cells if test_cells is not None else unit.test_cells

    if normalize == "author_minmax":
        Xnorm = normalize_cells(tr_cells + te_cells, "author_minmax")
        cycle_mode = "author_minmax"   # reference protocol: the cycle index is per-cell min-max too
    else:  # strict mode: statistics from source cells only, fixed-scale cycle (history-only)
        Xnorm = normalize_cells(tr_cells + te_cells, "source",
                                source_cells=tr_cells)
        cycle_mode = "scaled200"

    X_tr, y_tr, _ = assemble_matrix(tr_cells, Xnorm, view, cycle_mode)
    X_te, y_te, te_ids = assemble_matrix(te_cells, Xnorm, view, cycle_mode)
    X_tr, y_tr = subsample_context(X_tr, y_tr, context_cap, seed)

    t0 = time.time()
    pred = BACKBONES[backbone](X_tr, y_tr, X_te, seed)
    runtime = time.time() - t0

    overall = float(np.sqrt(np.mean((y_te - pred) ** 2)))
    per_cell = {cid: float(np.sqrt(np.mean((y_te[te_ids == cid]
                                            - pred[te_ids == cid]) ** 2)))
                for cid in np.unique(te_ids)}
    return {
        "unit": unit.unit_id, "view": view, "backbone": backbone,
        "seed": seed, "normalize": normalize,
        "n_train_rows": int(len(y_tr)), "n_test_rows": int(len(y_te)),
        "n_train_cells": len(tr_cells), "n_test_cells": len(te_cells),
        "rmse_overall": overall,
        "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
        "per_cell": per_cell,
        "runtime_s": runtime,
    }
