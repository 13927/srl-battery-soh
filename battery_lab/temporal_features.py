"""History-only cross-cycle features: the four families of transforms.

For every sensor feature z (the 16 statistics of the reference release):
  lag1       value at the previous cycle (first cycle initialised with itself)
  diff1      z_t - lag1
  from_first z_t - z_{t0}
  from_roll5 z_t - mean(z_{max(t0,t-4)..t})   # includes the current cycle, window <=5

Base features   = 16 statistics + cycle_scaled_200 -> 17 dimensions
History features = 17 + 16*4 = 81 dimensions

History-only is enforced: only cycles <=t are used, no backward fill, and every
transform stays inside a single cell.
"""

from __future__ import annotations

import numpy as np


def cycle_scaled(n_cycles: int, scale: float = 200.0) -> np.ndarray:
    """Cycle-position feature: cycle_index / 200 (0-based, fixed scale, not
    normalised by total life).

    Note: `n_cycles` is the number of **retained records** (the sequence length
    after the reader's row deletion), not the original row number in the released
    file -- see the definition of t in the paper, Section 2.2.
    """
    return (np.arange(n_cycles) / scale).reshape(-1, 1)


def cycle_author_minmax(n_cycles: int) -> np.ndarray:
    """Reference convention: the cycle index is min-max scaled to [-1, 1] by the
    cell's own full lifetime.

    Note: this reads the cell's complete lifetime length, so it is only for
    like-for-like comparison under the reference protocol; strict
    leave-one-cell-out must use cycle_scaled instead.
    Likewise, `n_cycles` is the number of **retained records**, not a released-file
    row number.
    """
    ci = np.arange(n_cycles, dtype=float)
    rng = max(ci.max() - ci.min(), 1.0)
    return (2 * (ci - ci.min()) / rng - 1).reshape(-1, 1)


def add_history_features(Z: np.ndarray) -> np.ndarray:
    """Take an (n_cycles, d) sensor matrix, return the (n_cycles, 4d) expansion.

    Strictly history-only: row t depends on Z[:t+1] alone.
    """
    n, d = Z.shape
    lag1 = np.vstack([Z[:1], Z[:-1]])            # first cycle: lag = itself
    diff1 = Z - lag1                              # first cycle: diff = 0
    from_first = Z - Z[:1]                        # offset from the first retained cycle

    # rolling mean (window <=5, current cycle included): cumulative sums, so no
    # future row can enter
    csum = np.cumsum(Z, axis=0)
    roll5 = np.empty_like(Z)
    for t in range(n):
        lo = max(0, t - 4)
        total = csum[t] - (csum[lo - 1] if lo > 0 else 0)
        roll5[t] = total / (t - lo + 1)
    from_roll5 = Z - roll5

    return np.hstack([lag1, diff1, from_first, from_roll5])


def build_view(Z: np.ndarray, view: str, cycle_mode: str = "scaled200") -> np.ndarray:
    """Build the full feature matrix for the raw / history views.

    raw:     [16 statistics, cycle feature] = 17 dimensions
    history: [16 statistics, cycle feature, 4x16 expansion] = 81 dimensions

    cycle_mode:
      scaled200      cycle/200, history-only (strict mode)
      author_minmax  per-cell full-lifetime min-max (reference protocol only)
    """
    n = len(Z)
    cyc = (cycle_scaled(n) if cycle_mode == "scaled200"
           else cycle_author_minmax(n))
    base = np.hstack([Z, cyc])
    if view == "raw":
        return base
    if view == "history":
        return np.hstack([base, add_history_features(Z)])
    raise ValueError(f"unknown view {view!r}")
