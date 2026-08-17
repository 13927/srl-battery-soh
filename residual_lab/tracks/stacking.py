"""Track A -- residuals in function space: residual stacking.

    y_hat = base(x) + alpha * corrector_residual(x)

Regression: fit the numeric residual r = y - y_hat_base directly.
Classification: correct the residual in logit space -- the corrector regresses
     the per-class logit error of the base model, and the final logits are
     base_logits + alpha * correction.

Both directions are supported:
  tabpfn->gbdt: TabPFN predicts first, GBDT fits its residual
  gbdt->tabpfn: GBDT predicts first, TabPFN fits its residual
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.model_selection import KFold

from residual_lab.data import DatasetSpec
from residual_lab.models import make_model

_EPS = 1e-7


def _to_logits(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, _EPS, 1 - _EPS))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _oof_proba(model_name: str, X, y, seed: int, n_folds: int = 5) -> np.ndarray:
    """Out-of-fold probabilities, so the corrector does not fit residuals the base
    model has already overfitted on its training rows."""
    n_classes = len(np.unique(y))
    oof = np.zeros((len(X), n_classes))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(X):
        m = make_model(model_name, seed=seed)
        m.fit(X[tr_idx], y[tr_idx])
        proba = m.predict_proba(X[va_idx])
        # a fold may miss a class: align the model classes_ to the global columns
        cols = np.asarray(m.classes_, dtype=int)
        oof[np.ix_(va_idx, cols)] = proba
    return oof


def _oof_pred_reg(model_name: str, X, y, seed: int, n_folds: int = 5) -> np.ndarray:
    oof = np.zeros(len(X))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(X):
        m = make_model(model_name, seed=seed)
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
    return oof


def residual_stacking_classification(
    spec: DatasetSpec,
    seed: int,
    base: str = "tabpfn",
    corrector: str = "gbdt_reg",
    alpha: float = 1.0,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Classification residual stacking in logit space; the corrector must be a
    regressor."""
    X_tr, y_tr, X_te = spec.X_train, spec.y_train, spec.X_test
    n_classes = len(np.unique(y_tr))

    # stage 1: out-of-fold probabilities on the training rows, full-fit
    # probabilities on the test rows
    oof_proba = _oof_proba(base, X_tr, y_tr, seed, n_folds)
    base_model = make_model(base, seed=seed)
    base_model.fit(X_tr, y_tr)
    test_proba = base_model.predict_proba(X_te)

    # stage 2: the target logit residual approximates the ideal logit of the
    # one-hot label minus the base logit.
    # Equivalently, use the cross-entropy gradient: residual target =
    # onehot - proba (the negative softmax gradient)
    onehot = np.eye(n_classes)[np.asarray(y_tr, dtype=int)]
    residual_target = onehot - oof_proba  # (n_train, n_classes)

    corrections = np.zeros((len(X_te), n_classes))
    for c in range(n_classes):
        reg = make_model(corrector, seed=seed)
        reg.fit(X_tr, residual_target[:, c])
        corrections[:, c] = reg.predict(X_te)

    final_logits = _to_logits(test_proba) + alpha * corrections
    final_proba = _softmax(final_logits)
    y_pred = final_proba.argmax(axis=1)
    return {
        "y_pred": y_pred,
        "y_proba": final_proba,
        "extra": {
            "base": base, "corrector": corrector, "alpha": alpha,
            "base_acc": float((test_proba.argmax(axis=1) == spec.y_test).mean()),
            "correction_norm": float(np.abs(corrections).mean()),
        },
    }


def residual_stacking_regression(
    spec: DatasetSpec,
    seed: int,
    base: str = "tabpfn_reg",
    corrector: str = "gbdt_reg",
    alpha: float = 1.0,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Regression residual stacking: y_hat = base(x) + alpha *
    corrector(y - base_oof(x))."""
    X_tr, y_tr, X_te = spec.X_train, spec.y_train, spec.X_test

    oof_pred = _oof_pred_reg(base, X_tr, y_tr, seed, n_folds)
    base_model = make_model(base, seed=seed)
    base_model.fit(X_tr, y_tr)
    base_test = base_model.predict(X_te)

    residual = y_tr - oof_pred
    reg = make_model(corrector, seed=seed)
    reg.fit(X_tr, residual)
    correction = reg.predict(X_te)

    y_pred = base_test + alpha * correction
    base_rmse = float(np.sqrt(np.mean((spec.y_test - base_test) ** 2)))
    return {
        "y_pred": y_pred,
        "y_proba": None,
        "extra": {
            "base": base, "corrector": corrector, "alpha": alpha,
            "base_rmse": base_rmse,
            "residual_std_train": float(residual.std()),
        },
    }


def make_stacking_method(base: str, corrector: str = "gbdt_reg", alpha: float = 1.0):
    """Build the closure the runner calls, dispatching on the task type."""

    def fn(spec: DatasetSpec, seed: int) -> Dict[str, Any]:
        if spec.task == "classification":
            return residual_stacking_classification(
                spec, seed, base=base, corrector=corrector, alpha=alpha
            )
        return residual_stacking_regression(
            spec, seed, base=base, corrector=corrector, alpha=alpha
        )

    return fn
