"""Dataset loading and noise injection.

CPU-friendly by design: the default datasets all have <=1000 training rows,
the local sklearn datasets need no download, and the OpenML ones are optional
(they require a network).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSpec:
    """One dataset slice for a single experiment."""

    name: str
    task: str  # "classification" | "regression"
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    noise_rate: float = 0.0
    seed: int = 0

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_classes(self) -> int:
        if self.task != "classification":
            return 0
        return len(np.unique(np.concatenate([self.y_train, self.y_test])))


def _make_synthetic_hard():
    """A locally generated hard classification set (no download): 15k x 40, 6
    classes, 5 per cent intrinsic label noise.

    At a context budget of 500 the TabPFN baseline is far from the ceiling, which
    is what the residual-scaling experiment needs.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=15000, n_features=40, n_informative=20, n_redundant=10,
        n_classes=6, n_clusters_per_class=2, class_sep=0.8, flip_y=0.05,
        random_state=42,
    )
    from sklearn.utils import Bunch

    return Bunch(data=X, target=y)


def _fetch_openml_bunch(data_id: int):
    """Load an OpenML dataset (network on first use, then cached under
    ~/scikit_learn_data).

    Returned as Bunch(data=float array, target=int array); NaNs are filled with
    column medians.
    """
    from sklearn.utils import Bunch

    d = skd.fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    X = np.asarray(d.data, dtype=np.float64)
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), med, X)
    y = np.searchsorted(np.unique(d.target), d.target)
    return Bunch(data=X, target=y)


# name -> (loader, task)
_LOCAL_DATASETS = {
    "breast_cancer": (skd.load_breast_cancer, "classification"),  # 569 x 30, 2 classes
    "wine": (skd.load_wine, "classification"),                    # 178 x 13, 3 classes
    "digits": (skd.load_digits, "classification"),                # 1797 x 64, 10 classes
    "iris": (skd.load_iris, "classification"),                    # 150 x 4, 3 classes
    "diabetes": (skd.load_diabetes, "regression"),                # 442 x 10
    # first call downloads ~1MB, then caches under ~/scikit_learn_data
    "california_housing": (skd.fetch_california_housing, "regression"),  # 20640 x 8
    # large hard set (~11MB download): 581k x 54, 7 classes; at a 500-row context
    # the baseline is far from the ceiling
    "covtype": (skd.fetch_covtype, "classification"),
    # locally synthesised hard set (no download): 15k x 40, 6 classes, 5 per cent
    # intrinsic noise
    "synthetic_hard": (_make_synthetic_hard, "classification"),
    # ---- the large-scale weak zone reported for TabPFN (N x d > 1e6, network on
    # first use) ----
    # source: "A Closer Look at TabPFN v2" (NeurIPS 2025), Table 1,
    # where TabPFN v2 averages rank 3.97, behind CatBoost (1.89) and RealMLP (1.94)
    "miniboone": (lambda: _fetch_openml_bunch(41150), "classification"),   # 130k x 50
    "jannis": (lambda: _fetch_openml_bunch(41168), "classification"),      # 84k x 54, 4 classes
    "higgs": (lambda: _fetch_openml_bunch(23512), "classification"),       # 98k x 28
}


def list_datasets(task: Optional[str] = None) -> list:
    names = list(_LOCAL_DATASETS.keys())
    if task is not None:
        names = [n for n in names if _LOCAL_DATASETS[n][1] == task]
    return names


def load_dataset(
    name: str,
    seed: int = 0,
    max_train: int = 500,
    max_test: int = 1000,
    test_size: float = 0.3,
    noise_rate: float = 0.0,
) -> DatasetSpec:
    """Load a dataset and split it. With noise_rate>0, symmetric noise is injected
    into the training labels.

    max_train/max_test cap the size so that TabPFN stays usable on CPU (they also
    serve the small-sample ablations).
    """
    if name not in _LOCAL_DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {list_datasets()}")
    loader, task = _LOCAL_DATASETS[name]
    bunch = loader()
    X, y = np.asarray(bunch.data, dtype=np.float64), np.asarray(bunch.target)
    if task == "classification":
        # remap labels to 0..K-1 (covtype and others are not 0-based)
        y = np.searchsorted(np.unique(y), y)

    stratify = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    rng = np.random.default_rng(seed)
    if len(X_tr) > max_train:
        idx = rng.choice(len(X_tr), size=max_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    if len(X_te) > max_test:
        idx = rng.choice(len(X_te), size=max_test, replace=False)
        X_te, y_te = X_te[idx], y_te[idx]

    if noise_rate > 0:
        if task == "classification":
            y_tr = inject_label_noise(y_tr, noise_rate, seed=seed)
        else:
            y_tr = inject_target_noise(y_tr, noise_rate, seed=seed)

    return DatasetSpec(
        name=name, task=task,
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        noise_rate=noise_rate, seed=seed,
    )


def inject_label_noise(y: np.ndarray, rate: float, seed: int = 0) -> np.ndarray:
    """Symmetric label noise: pick a fraction `rate` of rows at random and flip
    each label to another class, uniformly."""
    if not 0 < rate < 1:
        raise ValueError(f"noise rate must lie in (0,1), got {rate}")
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    classes = np.unique(y)
    n_flip = int(round(len(y) * rate))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    for i in flip_idx:
        others = classes[classes != y_noisy[i]]
        y_noisy[i] = rng.choice(others)
    return y_noisy


def inject_target_noise(y: np.ndarray, rate: float, seed: int = 0) -> np.ndarray:
    """Regression target noise: add a 2-sigma Gaussian perturbation to a fraction
    `rate` of the rows."""
    rng = np.random.default_rng(seed)
    y_noisy = y.astype(np.float64).copy()
    n_corrupt = int(round(len(y) * rate))
    idx = rng.choice(len(y), size=n_corrupt, replace=False)
    y_noisy[idx] += rng.normal(0, 2 * y.std(), size=n_corrupt)
    return y_noisy
