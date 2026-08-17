"""Evaluation metrics: one entry point for classification and regression."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    if y_proba is not None:
        labels = np.arange(y_proba.shape[1])
        try:
            if y_proba.shape[1] == 2:
                out["auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                out["auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", labels=labels
                )
        except ValueError:
            pass  # skip AUC when a class is missing from the test split
        out["log_loss"] = log_loss(y_true, y_proba, labels=labels)
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def evaluate(task: str, y_true, y_pred, y_proba=None) -> Dict[str, float]:
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_proba)
    return regression_metrics(y_true, y_pred)


# primary metric per task for comparisons and tests (larger is always better)
PRIMARY_METRIC = {"classification": "accuracy", "regression": "r2"}
