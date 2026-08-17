"""Borrowing from PINN/QPINN: physics priors as training-free post-processing and
as a fixed feature expansion.

The insight: what the PDE and monotonicity constraints of a PINN actually do is
make the output smooth and monotone along cycles. True SOH degrades smoothly and
monotonically while per-cycle prediction errors are close to independent noise,
so projecting along the trajectory removes noise -- with no gradients at all.

Two families:
  A. trajectory post-processing (applied to a frozen model's predictions)
     iso        non-increasing isotonic regression (whole test trajectory,
                transductive within a cell)
     smooth     rolling median
     iso_smooth smooth first, then enforce monotonicity
     causal     running minimum (uses predictions up to t only, fully inductive
                and deployable)
     causal_sm  causal rolling mean plus running minimum
  B. fixed kernel feature expansion (the classical counterpart of the QPINN
     quantum feature map)
     Nystroem RBF landmark map, untrainable, concatenated to the features before
     TabPFN

No test label is ever used; the cycle index is a feature, not a label, so using
it is legal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# A. trajectory post-processing
# ---------------------------------------------------------------------------

def _isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    """Non-increasing isotonic regression (PAVA): fit a non-decreasing regression
    to -y and negate."""
    from sklearn.isotonic import IsotonicRegression
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    return ir.fit_transform(x, y)


def _rolling_median(y: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return y.copy()
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    return np.array([np.median(yp[i:i + w]) for i in range(len(y))])


def _causal_cummin(y: np.ndarray) -> np.ndarray:
    """Causal monotonicity: SOH_hat(t) = min_{s<=t} pred(s). Past only, so it can
    run in service."""
    return np.minimum.accumulate(y)


def _causal_mean(y: np.ndarray, w: int = 5) -> np.ndarray:
    """Causal rolling mean (looks backwards only)."""
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = y[max(0, i - w + 1):i + 1].mean()
    return out


def _causal_isotonic(y: np.ndarray) -> np.ndarray:
    """Causal isotonic regression: at each t, project [0..t] to a non-increasing
    sequence and keep the last value.

    Unlike the running minimum this is the correct monotone projection in the L2
    sense (no systematic downward bias), which makes it possible to separate
    "variance reduction from monotonicity" from "bias correction from shifting
    the level down".
    """
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = _isotonic_decreasing(y[:i + 1])[-1]
    return out


def _shift_control(y: np.ndarray) -> np.ndarray:
    """Diagnostic control: shift the level down by the same constant amount as the
    running minimum does.

    If this matches the gain of the running minimum, that gain came from bias
    correction rather than from monotonicity.
    """
    return y - float(np.mean(y - _causal_cummin(y)))


POSTPROC = {
    "none": lambda y: y,
    "iso": _isotonic_decreasing,
    "smooth": lambda y: _rolling_median(y, 5),
    "iso_smooth": lambda y: _isotonic_decreasing(_rolling_median(y, 5)),
    "causal": _causal_cummin,
    "causal_sm": lambda y: _causal_cummin(_causal_mean(y, 5)),
    "iso_causal": _causal_isotonic,
    "shift_ctrl": _shift_control,
}


def apply_postproc(pred: np.ndarray, cell_ids: np.ndarray, mode: str,
                   cycle_order: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply the post-processing cell by cell, in cycle order.

    cycle_order: the within-cell cycle index of each row; None means the rows are
    already ordered by cycle (which holds in this project).
    """
    fn = POSTPROC[mode]
    out = pred.copy()
    for c in np.unique(cell_ids):
        m = np.flatnonzero(cell_ids == c)
        if cycle_order is not None:
            m = m[np.argsort(cycle_order[m])]
        out[m] = fn(pred[m])
    return out


# ---------------------------------------------------------------------------
# B. fixed kernel feature expansion (classical counterpart of the QPINN map)
# ---------------------------------------------------------------------------

def nystroem_expand(X_tr: np.ndarray, X_te: np.ndarray, n_components: int = 64,
                    seed: int = 0, concat: bool = True):
    """Nystroem RBF feature map; landmarks are fitted on the training side only.

    QPINN uses 8 qubits -> a 256-dimensional Hilbert space with M=256 landmarks.
    Here the classical RBF Nystroem map is used with a tunable width (TabPFN has
    a practical limit on feature count, so 64 is the safer default).
    """
    from sklearn.kernel_approximation import Nystroem
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_tr)
    Zt, Ze = sc.transform(X_tr), sc.transform(X_te)
    ny = Nystroem(kernel="rbf", gamma=1.0 / max(Zt.shape[1], 1),
                  n_components=min(n_components, len(Zt)), random_state=seed)
    ny.fit(Zt)
    Ft, Fe = ny.transform(Zt), ny.transform(Ze)
    if concat:
        return np.hstack([X_tr, Ft]), np.hstack([X_te, Fe])
    return Ft, Fe
