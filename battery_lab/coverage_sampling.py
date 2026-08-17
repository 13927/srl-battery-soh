"""Coverage sampling: test whether a context needs to *cover* diversity rather
than to *resemble* the target.

Motivation (the one positive lead from bexp10):
  early-only sampling is clearly worse (p=0.021), i.e. missing late-degradation
  rows hurts -> hypothesis: what matters is coverage (SOH range, degradation
  stage, cell diversity), not similarity (kNN/clustering, all of which failed)

Strategies (same budget, strict leave-one-cell-out):
  random          random rows (baseline, and the current winner)
  soh_strat       bin by SOH value, sample equally per bin (covers the label
                  space)
  stage_strat     bin by within-cell cycle position quantile, equally per bin
                  (covers degradation stages)
  cell_balanced   equal number of rows per cell (covers cell diversity)
  soh_cell        SOH stratification plus cell balancing (both at once)

Note that soh_strat uses the SOH labels of the *source* cells: those are
training-side labels and therefore legal (no target-cell label is involved).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from battery_lab.data_adapters import FitUnit, normalize_cells
from battery_lab.temporal_features import build_view


def _stack(cells, Xn, view):
    xs, ys, cids, cyc = [], [], [], []
    for c in cells:
        xs.append(build_view(Xn[c.cell_id], view, cycle_mode="scaled200"))
        ys.append(c.y)
        cids.extend([c.cell_id] * len(c.y))
        # within-cell cycle-position quantile (0..1), a proxy for the stage
        n = len(c.y)
        cyc.extend(np.arange(n) / max(n - 1, 1))
    return (np.vstack(xs), np.concatenate(ys), np.array(cids),
            np.array(cyc))


def _quota_sample(groups: List[np.ndarray], cap: int, rng) -> np.ndarray:
    """Split the quota evenly across groups, sample at random inside each, and
    top up from the remainder when a group is too small."""
    groups = [g for g in groups if len(g) > 0]
    if not groups:
        return np.array([], dtype=int)
    per = max(1, cap // len(groups))
    picked, leftover = [], []
    for g in groups:
        if len(g) <= per:
            picked.append(g)
        else:
            sel = rng.choice(g, per, replace=False)
            picked.append(sel)
            leftover.append(np.setdiff1d(g, sel))
    idx = np.concatenate(picked)
    if len(idx) < cap and leftover:
        pool = np.concatenate(leftover)
        need = min(cap - len(idx), len(pool))
        idx = np.concatenate([idx, rng.choice(pool, need, replace=False)])
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    return idx


def coverage_sample(y, cids, stage, mode: str, cap: int, seed: int,
                    n_bins: int = 10) -> np.ndarray:
    """Return the row indices of the context. Source-side information only."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n <= cap:
        return np.arange(n)

    if mode == "random":
        return rng.choice(n, cap, replace=False)

    if mode == "soh_strat":                       # cover the label range
        edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        groups = [np.flatnonzero((y >= edges[i]) & (y < edges[i + 1]))
                  for i in range(n_bins)]
        return _quota_sample(groups, cap, rng)

    if mode == "stage_strat":                     # cover degradation stages
        edges = np.linspace(0, 1 + 1e-9, n_bins + 1)
        groups = [np.flatnonzero((stage >= edges[i]) & (stage < edges[i + 1]))
                  for i in range(n_bins)]
        return _quota_sample(groups, cap, rng)

    if mode == "cell_balanced":                   # cover cell diversity
        groups = [np.flatnonzero(cids == c) for c in np.unique(cids)]
        return _quota_sample(groups, cap, rng)

    if mode == "soh_cell":                        # cover SOH x cell jointly
        edges = np.quantile(y, np.linspace(0, 1, 5 + 1))
        edges[-1] += 1e-9
        groups = []
        for c in np.unique(cids):
            m = cids == c
            for i in range(5):
                groups.append(np.flatnonzero(
                    m & (y >= edges[i]) & (y < edges[i + 1])))
        return _quota_sample(groups, cap, rng)

    raise ValueError(f"unknown coverage strategy {mode!r}")


def coverage_diagnostics(y_sel, y_all, cids_sel, cids_all,
                         stage_sel) -> Dict[str, float]:
    """Quantify how well the chosen context covers the pool (for the mechanism
    analysis)."""
    # SOH coverage: 1-Wasserstein distance between the SOH distribution of the
    # selection and that of the full pool (smaller = better covered)
    q = np.linspace(0, 1, 21)
    w_soh = float(np.mean(np.abs(np.quantile(y_sel, q) - np.quantile(y_all, q))))
    return {
        "soh_wasserstein": w_soh,
        "soh_range_ratio": float((y_sel.max() - y_sel.min())
                                 / (y_all.max() - y_all.min() + 1e-9)),
        "cell_coverage": float(len(np.unique(cids_sel))
                               / len(np.unique(cids_all))),
        "stage_std": float(np.std(stage_sel)),
    }


def coverage_comparison(unit: FitUnit, view: str, seed: int, cap: int = 8192,
                        modes=("random", "soh_strat", "stage_strat",
                               "cell_balanced", "soh_cell")) -> List[Dict[str, Any]]:
    from residual_lab.models import make_model

    tr, te = unit.train_cells, unit.test_cells
    Xn = normalize_cells(tr + te, "source", source_cells=tr)
    X_tr, y_tr, cid_tr, stage_tr = _stack(tr, Xn, view)
    X_te, y_te, cid_te, _ = _stack(te, Xn, view)

    out = []
    for mode in modes:
        idx = coverage_sample(y_tr, cid_tr, stage_tr, mode, cap, seed)
        m = make_model("tabpfn_reg", seed=seed)
        m.fit(X_tr[idx], y_tr[idx])
        pred = m.predict(X_te)
        per_cell = {c: float(np.sqrt(np.mean((y_te[cid_te == c]
                                             - pred[cid_te == c]) ** 2)))
                    for c in np.unique(cid_te)}
        diag = coverage_diagnostics(y_tr[idx], y_tr, cid_tr[idx], cid_tr,
                                    stage_tr[idx])
        out.append({"unit": unit.unit_id, "view": view, "mode": mode,
                    "cap": cap, "seed": seed, "n_ctx": int(len(idx)),
                    "rmse_cell_macro": float(np.mean(list(per_cell.values()))),
                    "rmse_overall": float(np.sqrt(np.mean((y_te - pred) ** 2))),
                    **diag})
    return out
