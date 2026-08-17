"""residual_lab: shared model/metric/runner helpers used by some experiments.

Provides the dataset loaders, model factory, metrics and paired-statistics
helpers that the battery experiments reuse (tracks.stacking and
tracks.context_boosting are the two residual constructions still referenced).
"""

from residual_lab.data import DatasetSpec, load_dataset, inject_label_noise, list_datasets
from residual_lab.models import make_model, list_models
from residual_lab.runner import ExperimentRunner, RunRecord

__all__ = [
    "DatasetSpec",
    "load_dataset",
    "inject_label_noise",
    "list_datasets",
    "make_model",
    "list_models",
    "ExperimentRunner",
    "RunRecord",
]
