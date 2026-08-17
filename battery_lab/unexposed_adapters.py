"""Adapter for the unexposed library: strict base features from raw
charge/discharge curves.

Feature layout (MATR style):
  fixed time points 60/120/180/240/300 s x {current, voltage, temperature} = 15
  early-window statistics of the three modalities (mean/std/slope each)     = 9
  operating condition and cycle position (cycle_scaled_200 + 3 constants)   = 4
  28 strict base features in total

Missing values are filled history-only:
  1. the fill rule is fitted on the source group (column medians)
  2. a target stream may only be forward-filled from its own history; backward
     fill is forbidden
  3. where no history exists the source-group statistic is used and a
     missingness indicator is set
  4. whether a fill happened is recorded per target and per column

MATR-CLO (2019-01-24, the closed-loop-optimisation batch of Attia et al. 2020)
is not among the three MIT batches used by PINN4SOH
(2017-05-12/2017-06-30/2018-04-12), nor in the MATR Batch 1 used during
development, which is why it serves as the unexposed library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from battery_lab.data_adapters import CellData, FitUnit

MATR_TIMEPOINTS = (60.0, 120.0, 180.0, 240.0, 300.0)
EARLY_WINDOW_S = 300.0
STRICT_DIM = 28


def _interp_at(t: np.ndarray, v: np.ndarray, times) -> np.ndarray:
    """Linear interpolation at the fixed time points (inside the observed window
    only)."""
    if len(t) < 2:
        return np.full(len(times), np.nan)
    return np.interp(times, t, v, left=v[0], right=np.nan)


def _early_stats(t: np.ndarray, v: np.ndarray) -> Tuple[float, float, float]:
    """Early-window statistics: mean, standard deviation, linear slope in time."""
    if len(t) < 2:
        return (np.nan, np.nan, np.nan)
    slope = np.polyfit(t, v, 1)[0] if np.ptp(t) > 0 else 0.0
    return (float(np.mean(v)), float(np.std(v)), float(slope))


def strict_features_from_cycle(
    t: np.ndarray, I: np.ndarray, V: np.ndarray, T: np.ndarray,
    cycle_index: int, conditions: Tuple[float, float, float],
) -> np.ndarray:
    """One cycle -> 28 strict base features (only the first EARLY_WINDOW_S seconds
    are read)."""
    t = np.asarray(t, dtype=float)
    t = t - t[0] if len(t) else t
    mask = t <= EARLY_WINDOW_S
    tw, Iw, Vw, Tw = t[mask], I[mask], V[mask], T[mask]

    feats: List[float] = []
    # 15: fixed time points x three modalities
    for arr in (Iw, Vw, Tw):
        feats.extend(_interp_at(tw, arr, MATR_TIMEPOINTS))
    # 9: early-window statistics of the three modalities
    for arr in (Iw, Vw, Tw):
        feats.extend(_early_stats(tw, arr))
    # 4: cycle position + three operating-condition constants
    feats.append(cycle_index / 200.0)
    feats.extend(conditions)
    out = np.asarray(feats, dtype=float)
    assert out.shape == (STRICT_DIM,), f"width {out.shape} != {STRICT_DIM}"
    return out


# ---------------------------------------------------------------------------
# history-only imputation
# ---------------------------------------------------------------------------

@dataclass
class ImputationRule:
    """Fill rule fitted on the source group (per-column medians)."""

    col_median: np.ndarray
    n_cols: int

    @classmethod
    def fit(cls, X_source: np.ndarray) -> "ImputationRule":
        med = np.nanmedian(X_source, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        return cls(col_median=med, n_cols=X_source.shape[1])


def impute_history_only(
    X: np.ndarray, rule: ImputationRule,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Fill a target stream: forward fill from its own history, falling back to the
    source-group medians where no history exists.

    Returns (filled features, missingness indicators, per-column fill counts).
    Backward fill is forbidden -- no later cycle is ever read.
    """
    X_ff, still_nan, miss = precompute_ffill(X)
    X_out = apply_rule(X_ff, still_nan, rule)
    counts = {f"col{j}": int(miss[:, j].sum())
              for j in range(X.shape[1]) if miss[:, j].any()}
    return X_out, miss.astype(float), counts


def precompute_ffill(X: np.ndarray):
    """Forward-fill within a cell once (independent of the source group, so the
    result can be cached and reused).

    Returns (forward-filled matrix, still-missing mask, original missing mask).
    A still-missing entry means the column had no observation before it (a leading
    gap).
    """
    X = np.array(X, dtype=float, copy=True)
    miss = ~np.isfinite(X)
    n, d = X.shape
    for j in range(d):
        col = X[:, j]
        valid = np.isfinite(col)
        if valid.all():
            continue
        # vectorised forward fill: take the most recent valid index
        idx = np.where(valid, np.arange(n), -1)
        idx = np.maximum.accumulate(idx)
        take = idx >= 0
        col[take] = col[idx[take]]
        X[:, j] = col
    still_nan = ~np.isfinite(X)
    return X, still_nan, miss


def apply_rule(X_ff: np.ndarray, still_nan: np.ndarray,
               rule: ImputationRule) -> np.ndarray:
    """Apply the source-group rule to the leading gaps (vectorised, O(nd) per
    fold)."""
    if not still_nan.any():
        return X_ff
    X = X_ff.copy()
    med = np.broadcast_to(rule.col_median, X.shape)
    X[still_nan] = med[still_nan]
    assert np.isfinite(X).all(), "non-finite values remain after imputation"
    return X


# ---------------------------------------------------------------------------
# MATR-CLO loading
# ---------------------------------------------------------------------------

def load_matr_clo(
    mat_path: Path, nominal_capacity: float = 1.1,
    min_cycles: int = 30, max_cells: Optional[int] = None,
) -> FitUnit:
    """Load the MATR-CLO batch from MATLAB v7.3 (.mat) into a FitUnit.

    SOH = discharge capacity / 1.1 Ah nominal (as for MIT/HUST in PINN4SOH).
    Cell split: sorted index % 5 == 0 -> test (the equivalent of the reference MIT
    rule).
    """
    import h5py

    cells: List[CellData] = []
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        n_cells = batch["summary"].shape[0]
        for i in range(n_cells):
            if max_cells and len(cells) >= max_cells:
                break
            # summary: QDischarge per cycle
            summ = f[batch["summary"][i, 0]]
            Qd = np.array(summ["QDischarge"]).flatten()
            cyc_life = len(Qd)
            if cyc_life < min_cycles:
                continue
            cyc_ref = f[batch["cycles"][i, 0]]

            rows, ys = [], []
            for c in range(cyc_life):
                try:
                    t = np.array(f[cyc_ref["t"][c, 0]]).flatten() * 60.0  # minutes -> seconds
                    I = np.array(f[cyc_ref["I"][c, 0]]).flatten()
                    V = np.array(f[cyc_ref["V"][c, 0]]).flatten()
                    Tt = np.array(f[cyc_ref["T"][c, 0]]).flatten()
                except (KeyError, IndexError, ValueError):
                    continue
                if len(t) < 5:
                    continue
                soh = float(Qd[c]) / nominal_capacity
                if not (0.3 < soh < 1.3):
                    continue
                rows.append(strict_features_from_cycle(
                    t, I, V, Tt, cycle_index=c,
                    conditions=(30.0, 2.0, float(np.nanmean(np.abs(I))))))
                ys.append(soh)
            if len(rows) < min_cycles:
                continue
            cid = f"CLO_cell{i:02d}"
            cells.append(CellData(cell_id=cid, X=np.vstack(rows),
                                  y=np.asarray(ys)))

    # split: index % 5 == 0 -> test
    for k, c in enumerate(cells):
        c.is_test = (k + 1) % 5 == 0
    return FitUnit(unit_id="MATR-CLO", dataset="MATR_CLO", cells=cells)
