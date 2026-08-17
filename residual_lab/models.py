"""Model adapters.

One interface: every model is an sklearn-style estimator
(fit/predict/predict_proba). TabPFN is imported lazily, so the rest of the
package (including the smoke tests) works without it installed.

Registry keys:
  classification: tabpfn, gbdt, rf, logreg, knn
  regression:     tabpfn_reg, gbdt_reg, rf_reg, ridge
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier

# recommended CPU settings for TabPFN: few estimators, silence the pretraining
# size warning
_TABPFN_CPU_KW = dict(n_estimators=2, device="cpu", ignore_pretraining_limits=True)

# local TabPFN v3 weight directory (for offline use, avoids a runtime download).
# Override with the TABPFN_WEIGHTS_DIR environment variable.
TABPFN_WEIGHTS_DIR = Path(
    os.environ.get("TABPFN_WEIGHTS_DIR", "/Users/aaa/Downloads/tabpfn_3")
)
_TABPFN_CLF_CKPT = TABPFN_WEIGHTS_DIR / "tabpfn-v3-classifier-v3_default.ckpt"
_TABPFN_REG_CKPT = TABPFN_WEIGHTS_DIR / "tabpfn-v3-regressor-v3_default.ckpt"


def _tabpfn_model_path(ckpt: Path) -> str:
    """Use the local weights when present, otherwise fall back to the official
    default (which downloads them)."""
    return str(ckpt) if ckpt.exists() else "auto"


def _make_tabpfn_clf(seed: int) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        random_state=seed,
        model_path=_tabpfn_model_path(_TABPFN_CLF_CKPT),
        **_TABPFN_CPU_KW,
    )


def _make_tabpfn_reg(seed: int) -> Any:
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        random_state=seed,
        model_path=_tabpfn_model_path(_TABPFN_REG_CKPT),
        **_TABPFN_CPU_KW,
    )


_REGISTRY: Dict[str, Callable[[int], Any]] = {
    # classification
    "tabpfn": _make_tabpfn_clf,
    "gbdt": lambda seed: HistGradientBoostingClassifier(random_state=seed),
    "rf": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
    "logreg": lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    "knn": lambda seed: KNeighborsClassifier(),
    # regression
    "tabpfn_reg": _make_tabpfn_reg,
    "gbdt_reg": lambda seed: HistGradientBoostingRegressor(random_state=seed),
    "rf_reg": lambda seed: RandomForestRegressor(n_estimators=200, random_state=seed),
    "ridge": lambda seed: Ridge(),
}


def list_models() -> list:
    return list(_REGISTRY.keys())


def make_model(name: str, seed: int = 0, **overrides) -> Any:
    """Build a model by name. overrides are passed straight to the constructor
    (used for ablations)."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown model {name!r}; available: {list_models()}")
    model = _REGISTRY[name](seed)
    if overrides:
        model.set_params(**overrides)
    return model


def register_model(name: str, factory: Callable[[int], Any]) -> None:
    """Register a custom model (e.g. a composite method) so the runner can call it
    by name."""
    _REGISTRY[name] = factory


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401

        return True
    except ImportError:
        return False
