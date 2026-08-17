"""Conflict-gated constraint application.

Lineage:
  - Prior-data conflict checks (Evans & Moshonov 2006; Nott et al.):
    a statistic tests whether the prior conflicts with the data; on conflict the
    prior is dropped.
  - Learn-then-Test / Conformal Risk Control (Angelopoulos et al. 2021/2022):
    calibrate the post-processing parameter lambda on a calibration set so that
    the risk stays <= alpha with a distribution-free guarantee.
  - Selective prediction (El-Yaniv & Wiener 2010): the gate/abstain paradigm.

Method:
  For a shape constraint C (monotone iso / smoothing):
  1. conflict statistics, in three tiers ordered by deployment legality
     Tier-S: rebound ratio of the source-cell labels (unit level, verified in
             bexp12/13)
     Tier-T: violation statistics of the target cell's predicted trajectory
             (cell level, label-free, legal in deployment)
     Tier-O: statistics of the true target labels (analysis only, an upper bound)
  2. LTT gating: calibrate the threshold lambda* on the source cells (nested
     leave-one-cell-out) so that
     the risk of "applying the constraint does harm" is held at level alpha by a
     binomial test
  3. deployment: apply the constraint to a cell only when v_pred(i) <= lambda*
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from battery_lab.physics_posthoc import POSTPROC


# ---------------------------------------------------------------------------
# conflict statistics (all label-free; the input is one trajectory vector)
# ---------------------------------------------------------------------------

def violation_ratio(traj: np.ndarray) -> float:
    """Conflict statistic for the monotone prior: sum of positive differences over
    sum of negative differences.

    Called on a predicted trajectory this is Tier-T (label-free); called on a
    label trajectory it is Tier-S/O.
    """
    d = np.diff(np.asarray(traj, dtype=float))
    up = float(d[d > 0].sum())
    dn = float(-d[d < 0].sum())
    return up / max(dn, 1e-9)


def correction_magnitude(traj: np.ndarray, mode: str = "iso") -> float:
    """Magnitude of the correction made by the projection (a label-free auxiliary
    statistic)."""
    proj = POSTPROC[mode](np.asarray(traj, dtype=float))
    return float(np.mean(np.abs(traj - proj)))


def roughness_ratio(traj: np.ndarray, ref_rough: float) -> float:
    """Conflict statistic for the smoothing prior: roughness of the predicted
    trajectory over a reference roughness from the source labels.

    Roughness is the standard deviation of successive differences; ref_rough is
    computed from the source-cell labels.
    """
    d = np.diff(np.asarray(traj, dtype=float))
    return float(np.std(d)) / max(ref_rough, 1e-9)


def label_roughness(trajs: Sequence[np.ndarray]) -> float:
    """Reference roughness of the source-cell labels (median over cells)."""
    vals = [float(np.std(np.diff(np.asarray(t, dtype=float))))
            for t in trajs if len(t) > 2]
    return float(np.median(vals)) if vals else 1e-9


# ---------------------------------------------------------------------------
# LTT gate calibration (learn-then-test fixed sequence + binomial test)
# ---------------------------------------------------------------------------

def ltt_calibrate(
    stats: np.ndarray, harms: np.ndarray,
    alpha: float = 0.1, risk_target: float = 0.10,
    n_grid: int = 30,
) -> Dict[str, Any]:
    """Calibrate the gate threshold lambda*.

    stats: the conflict statistic v_pred of each calibration cell
    harms: whether applying the constraint did harm (bool, True if RMSE worsened)
    Guarantee: the selected lambda* is such that, over the calibration cells with
          {v <= lambda*}, the harm rate is significantly <= risk_target
          (one-sided binomial test p <= alpha; the LTT fixed sequence walks
          lambda upwards and stops at the first failure)

    Returns lambda* (possibly -inf, meaning "never apply") and the trace.
    """
    from scipy.stats import binomtest

    stats = np.asarray(stats, dtype=float)
    harms = np.asarray(harms, dtype=bool)
    grid = np.unique(np.quantile(stats, np.linspace(0, 1, n_grid)))
    lam_star = -np.inf
    trace = []
    for lam in grid:                       # fixed sequence: lambda ascending
        m = stats <= lam
        n, k = int(m.sum()), int(harms[m].sum())
        if n == 0:
            continue
        p = binomtest(k, n, risk_target, alternative="greater").pvalue
        ok = p > alpha                     # "risk <= target" not rejected -> accept
        trace.append({"lam": float(lam), "n": n, "harm": k, "p": float(p),
                      "accept": bool(ok)})
        if ok:
            lam_star = float(lam)
        else:
            break                          # fixed-sequence testing stops at the first failure
    return {"lam_star": float(lam_star), "alpha": alpha,
            "risk_target": risk_target, "trace": trace}


def apply_gate(pred: np.ndarray, cell_ids: np.ndarray, lam_star: float,
               mode: str = "iso") -> np.ndarray:
    """Gate the constraint cell by cell: project only when v_pred(i) <= lambda*.
    No target label is used."""
    out = pred.copy()
    for c in np.unique(cell_ids):
        m = np.flatnonzero(cell_ids == c)
        if violation_ratio(pred[m]) <= lam_star:
            out[m] = POSTPROC[mode](pred[m])
    return out


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------

def per_cell_record(pred, y, cell_ids, modes=("iso", "causal", "smooth",
                                              "shift_ctrl")) -> List[Dict[str, Any]]:
    """Turn one inference output into per-cell records (statistics plus the paired
    gain/loss of each constraint)."""
    rows = []
    for c in np.unique(cell_ids):
        m = np.flatnonzero(cell_ids == c)
        p, t = pred[m], y[m]
        base = float(np.sqrt(np.mean((t - p) ** 2)))
        rec = {
            "cell_id": str(c), "n_cycles": int(len(m)),
            "v_pred": violation_ratio(p),           # Tier-T
            "corr_mag": correction_magnitude(p),    # Tier-T auxiliary
            "v_true": violation_ratio(t),           # Tier-O (analysis only)
            "rough_pred": float(np.std(np.diff(p))) if len(p) > 2 else 0.0,
            "rough_true": float(np.std(np.diff(t))) if len(t) > 2 else 0.0,
            "rmse_none": base,
        }
        for mode in modes:
            q = POSTPROC[mode](p)
            rec[f"rmse_{mode}"] = float(np.sqrt(np.mean((t - q) ** 2)))
            rec[f"delta_{mode}"] = (rec[f"rmse_{mode}"] - base) / max(base, 1e-12) * 100
        rows.append(rec)
    return rows
