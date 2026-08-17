"""Track B -- residuals in data space: residual-driven context boosting.

The model stays fixed (TabPFN) while the context data D is refined iteratively.

  Round t: ŷ = model(X_test | D_t)
           the residual signal is the out-of-fold error inside D_t (no test label
           is read)
           D_{t+1} = Refine(D_t, residual)

Three refinement strategies:
  prune:    drop the rows with the most confident out-of-fold errors (suspected
            noise or harmful rows)
  relabel:  replace the label of consistently misclassified rows with the
            out-of-fold prediction (label correction)
  reweight: duplicate hard rows to raise their weight in the context (a discrete
            approximation of soft weighting)

The internal out-of-fold accuracy of D is recorded each round as a convergence
signal; the loop stops early once it no longer improves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import StratifiedKFold

from residual_lab.data import DatasetSpec
from residual_lab.models import make_model


@dataclass
class BoostingTrace:
    """Diagnostics of one iteration, for analysing convergence."""

    round_id: int
    n_context: int
    oof_accuracy: float
    n_pruned: int = 0
    n_relabeled: int = 0
    n_reweighted: int = 0


def _oof_signal(model_name: str, X, y, seed: int, n_folds: int = 3):
    """Out-of-fold prediction inside the context: returns (oof_pred, oof_conf,
    oof_proba).

    This is the source of the residual signal; no test label is needed.
    """
    n = len(X)
    classes = np.unique(y)
    n_classes = len(classes)
    oof_pred = np.zeros(n, dtype=y.dtype)
    oof_conf = np.zeros(n)
    n_folds = min(n_folds, int(np.bincount(np.searchsorted(classes, y)).min()))
    if n_folds < 2:
        return y.copy(), np.ones(n), None  # class too small for OOF: no signal
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        m = make_model(model_name, seed=seed)
        m.fit(X[tr_idx], y[tr_idx])
        proba = m.predict_proba(X[va_idx])
        local = proba.argmax(axis=1)
        oof_pred[va_idx] = np.asarray(m.classes_)[local]
        oof_conf[va_idx] = proba.max(axis=1)
    return oof_pred, oof_conf, None


def residual_context_boosting(
    spec: DatasetSpec,
    seed: int,
    model_name: str = "tabpfn",
    strategy: str = "prune",
    n_rounds: int = 3,
    prune_ratio: float = 0.1,
    relabel_conf: float = 0.85,
    min_context: int = 40,
) -> Dict[str, Any]:
    """Residual-driven iterative context refinement (classification only)."""
    if spec.task != "classification":
        raise ValueError("context boosting currently supports classification only")
    if strategy not in ("prune", "relabel", "reweight"):
        raise ValueError(f"unknown strategy {strategy!r}")

    X_ctx, y_ctx = spec.X_train.copy(), spec.y_train.copy()
    traces: List[BoostingTrace] = []
    best_oof, best_ctx = -1.0, (X_ctx, y_ctx)

    for t in range(n_rounds):
        oof_pred, oof_conf, _ = _oof_signal(model_name, X_ctx, y_ctx, seed + t)
        wrong = oof_pred != y_ctx
        oof_acc = float(1 - wrong.mean())
        trace = BoostingTrace(round_id=t, n_context=len(X_ctx), oof_accuracy=oof_acc)

        if oof_acc > best_oof:
            best_oof, best_ctx = oof_acc, (X_ctx.copy(), y_ctx.copy())

        # residual signal: rows predicted wrongly with high confidence are the
        # most suspicious
        suspicion = np.where(wrong, oof_conf, 0.0)

        if strategy == "prune":
            n_prune = min(int(len(X_ctx) * prune_ratio), len(X_ctx) - min_context)
            if n_prune <= 0 or suspicion.max() == 0:
                traces.append(trace)
                break
            drop = np.argsort(suspicion)[-n_prune:]
            drop = drop[suspicion[drop] > 0]  # drop only genuinely wrong rows
            keep = np.setdiff1d(np.arange(len(X_ctx)), drop)
            trace.n_pruned = len(drop)
            X_ctx, y_ctx = X_ctx[keep], y_ctx[keep]

        elif strategy == "relabel":
            flip = wrong & (oof_conf >= relabel_conf)
            trace.n_relabeled = int(flip.sum())
            if not flip.any():
                traces.append(trace)
                break
            y_ctx = y_ctx.copy()
            y_ctx[flip] = oof_pred[flip]

        elif strategy == "reweight":
            # duplicate correctly but weakly predicted "hard" rows to raise their
            # weight in the context
            hard = (~wrong) & (oof_conf < np.median(oof_conf))
            trace.n_reweighted = int(hard.sum())
            if not hard.any():
                traces.append(trace)
                break
            X_ctx = np.vstack([X_ctx, X_ctx[hard]])
            y_ctx = np.concatenate([y_ctx, y_ctx[hard]])

        traces.append(trace)

    # predict with the context that scored best out-of-fold (guards against
    # over-refinement)
    X_best, y_best = best_ctx
    final = make_model(model_name, seed=seed)
    final.fit(X_best, y_best)
    y_proba = final.predict_proba(spec.X_test)
    return {
        "y_pred": y_proba.argmax(axis=1),
        "y_proba": y_proba,
        "extra": {
            "strategy": strategy,
            "final_context_size": len(X_best),
            "rounds_run": len(traces),
            "trace": [vars(tr) for tr in traces],
        },
    }


def make_context_boosting_method(
    model_name: str = "tabpfn", strategy: str = "prune", **kwargs
):
    """Build the closure the runner calls."""

    def fn(spec: DatasetSpec, seed: int) -> Dict[str, Any]:
        return residual_context_boosting(
            spec, seed, model_name=model_name, strategy=strategy, **kwargs
        )

    return fn
