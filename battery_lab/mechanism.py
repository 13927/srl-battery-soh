"""Mechanism-level descriptors: when does the history view help? (association
only, no predictive claim is made.)

Computed per evaluation unit, always from source cells only, which keeps them as
legal as the view-selection rule itself:
  dim_row_ratio      history width / source training rows -- dimension dilution
  dynamic_share      share of "dynamic" base features
                     (slope/entropy/std/kurtosis/skewness)
  trend_strength     mean |rank correlation| between feature and cycle index --
                     strength of the degradation trend
  mean_cell_len      mean number of cycles per cell -- history available to the
                     cross-cycle expansion
  group_shift_auc    AUC of a source-versus-target group classifier --
                     distribution shift
  random_vs_group    R2 gap between a random row split and a per-cell split --
                     total leakage/shift
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats import spearmanr

from battery_lab.data_adapters import FEATURE_COLS, FitUnit, normalize_cells
from battery_lab.temporal_features import build_view

# the "dynamic" members of the 16 statistics (curve shape or change rather than
# absolute level)
DYNAMIC_KEYS = ("slope", "entropy", "std", "kurtosis", "skewness")


def dynamic_share() -> float:
    n_dyn = sum(any(k in c for k in DYNAMIC_KEYS) for c in FEATURE_COLS)
    return n_dyn / len(FEATURE_COLS)


def compute_indicators(unit: FitUnit, seed: int = 0) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, train_test_split

    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "source", source_cells=tr)

    # source-side row count and history width
    X_hist, y_src, ids = _stack(tr, Xn, "history")
    X_raw, _, _ = _stack(tr, Xn, "raw")
    dim_row_ratio = X_hist.shape[1] / len(X_hist)

    # trend strength: mean |Spearman| between each base feature and the cycle
    # index, averaged within source cells
    trends = []
    for c in tr:
        Z = Xn[c.cell_id]
        cyc = np.arange(len(Z))
        if len(Z) < 5:
            continue
        rs = [abs(spearmanr(Z[:, j], cyc).statistic) for j in range(Z.shape[1])]
        trends.append(np.nanmean(rs))
    trend_strength = float(np.nanmean(trends)) if trends else float("nan")

    # group shift: separability of source versus target (base features, row level)
    X_src_base, _, _ = _stack(tr, Xn, "raw")
    X_tgt_base, _, _ = _stack(te, Xn, "raw")
    Xg = np.vstack([X_src_base, X_tgt_base])
    yg = np.r_[np.zeros(len(X_src_base)), np.ones(len(X_tgt_base))]
    n_sub = min(4000, len(Xg))
    idx = np.random.default_rng(seed).choice(len(Xg), n_sub, replace=False)
    try:
        auc = float(np.mean(cross_val_score(
            HistGradientBoostingClassifier(max_iter=100, random_state=seed),
            Xg[idx], yg[idx], cv=3, scoring="roc_auc")))
    except ValueError:
        auc = float("nan")

    # random row split versus per-cell split (ridge on the raw view, inside the
    # source cells)
    rnd = _ridge_r2_random(X_src_base, y_src, seed)
    grp = _ridge_r2_by_cell(X_src_base, y_src, ids, seed)

    return {
        "unit": unit.unit_id,
        "n_source_cells": len(tr),
        "n_source_rows": int(len(X_hist)),
        "hist_dim": int(X_hist.shape[1]),
        "dim_row_ratio": float(dim_row_ratio),
        "dynamic_share": dynamic_share(),
        "trend_strength": trend_strength,
        "mean_cell_len": float(np.mean([len(c.y) for c in tr])),
        "group_shift_auc": auc,
        "r2_random_split": rnd,
        "r2_group_split": grp,
        "random_minus_group": (rnd - grp) if np.isfinite(rnd + grp) else float("nan"),
    }


def _stack(cells, Xn, view):
    xs, ys, ids = [], [], []
    for c in cells:
        xs.append(build_view(Xn[c.cell_id], view, cycle_mode="scaled200"))
        ys.append(c.y)
        ids.extend([c.cell_id] * len(c.y))
    return np.vstack(xs), np.concatenate(ys), np.array(ids)


def _ridge_r2_random(X, y, seed):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    return float(np.mean(cross_val_score(Ridge(), X, y, cv=3, scoring="r2")))


def _ridge_r2_by_cell(X, y, ids, seed):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_score
    n_groups = len(np.unique(ids))
    if n_groups < 3:
        return float("nan")
    return float(np.mean(cross_val_score(
        Ridge(), X, y, cv=GroupKFold(n_splits=min(3, n_groups)),
        groups=ids, scoring="r2")))
