"""Smoke tests for the framework: sklearn models only, no TabPFN, runs in seconds.

Run with: .venv/bin/python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from residual_lab.data import inject_label_noise, list_datasets, load_dataset
from residual_lab.runner import ExperimentRunner, sklearn_method
from residual_lab.stats import compare_methods, paired_wilcoxon
from residual_lab.tracks.context_boosting import make_context_boosting_method
from residual_lab.tracks.stacking import make_stacking_method


# ---------- data ----------

def test_load_dataset_shapes():
    spec = load_dataset("wine", seed=0, max_train=100)
    assert spec.task == "classification"
    assert spec.n_train <= 100
    assert len(spec.X_test) == len(spec.y_test)
    assert spec.n_classes == 3


def test_load_regression_dataset():
    spec = load_dataset("diabetes", seed=0, max_train=200)
    assert spec.task == "regression"


def test_noise_injection_rate():
    y = np.array([0, 1] * 100)
    y_noisy = inject_label_noise(y, rate=0.2, seed=0)
    flipped = (y != y_noisy).mean()
    assert abs(flipped - 0.2) < 0.02  # exactly 20 per cent flipped
    assert set(np.unique(y_noisy)) <= {0, 1}


def test_noise_deterministic():
    y = np.arange(100) % 3
    a = inject_label_noise(y, 0.3, seed=7)
    b = inject_label_noise(y, 0.3, seed=7)
    assert (a == b).all()


# ---------- runner ----------

def test_runner_matrix_and_resume(tmp_path):
    methods = {"gbdt": sklearn_method("gbdt"), "logreg": sklearn_method("logreg")}
    runner = ExperimentRunner(out_dir=str(tmp_path), exp_name="smoke")
    runner.run_matrix(["wine"], methods, seeds=[0], noise_rates=[0.0],
                      max_train=100, verbose=False)
    assert len(runner.records) == 2
    assert all(r.error is None for r in runner.records)
    assert all(r.primary_score > 0.5 for r in runner.records)

    # resume: configurations already finished are not run again
    runner2 = ExperimentRunner(out_dir=str(tmp_path), exp_name="smoke")
    runner2.run_matrix(["wine"], methods, seeds=[0], noise_rates=[0.0],
                       max_train=100, verbose=False)
    assert len(runner2.records) == 2

    csv_path = runner2.to_csv()
    assert csv_path.exists()
    assert "accuracy" in csv_path.read_text()


def test_scores_by_method_paired(tmp_path):
    methods = {"gbdt": sklearn_method("gbdt"), "knn": sklearn_method("knn")}
    runner = ExperimentRunner(out_dir=str(tmp_path), exp_name="paired")
    runner.run_matrix(["wine", "iris"], methods, seeds=[0, 1],
                      noise_rates=[0.0], max_train=100, verbose=False)
    scores = runner.scores_by_method()
    assert set(scores) == {"gbdt", "knn"}
    assert len(scores["gbdt"]) == len(scores["knn"]) == 4  # 2 datasets x 2 seeds


# ---------- stats ----------

def test_wilcoxon_direction():
    a = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
    b = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85]
    r = paired_wilcoxon(a, b)
    assert r["mean_diff"] > 0
    assert r["win_rate"] == 1.0
    assert r["p_value"] < 0.05


def test_compare_methods_requires_baseline():
    with pytest.raises(ValueError):
        compare_methods({"a": [1.0]}, baseline="missing")


# ---------- Track A: stacking (sklearn models stand in for TabPFN here) ----------

def test_stacking_classification_runs():
    spec = load_dataset("wine", seed=0, max_train=120)
    fn = make_stacking_method(base="logreg", corrector="rf_reg")
    out = fn(spec, seed=0)
    assert out["y_proba"].shape == (len(spec.X_test), 3)
    assert np.allclose(out["y_proba"].sum(axis=1), 1.0, atol=1e-6)
    acc = (out["y_pred"] == spec.y_test).mean()
    assert acc > 0.6


def test_stacking_regression_runs():
    spec = load_dataset("diabetes", seed=0, max_train=200)
    fn = make_stacking_method(base="ridge", corrector="gbdt_reg")
    out = fn(spec, seed=0)
    assert len(out["y_pred"]) == len(spec.y_test)
    assert "base_rmse" in out["extra"]


def test_stacking_alpha_zero_equals_base():
    """At alpha=0 stacking must reduce to the base model (residual correction off)."""
    spec = load_dataset("wine", seed=1, max_train=120)
    stacked = make_stacking_method(base="logreg", corrector="rf_reg", alpha=0.0)(spec, 1)
    base = sklearn_method("logreg")(spec, 1)
    assert (stacked["y_pred"] == base["y_pred"]).all()


# ---------- Track B: context boosting ----------

@pytest.mark.parametrize("strategy", ["prune", "relabel", "reweight"])
def test_context_boosting_strategies(strategy):
    spec = load_dataset("wine", seed=0, max_train=120, noise_rate=0.2)
    fn = make_context_boosting_method("gbdt", strategy=strategy, n_rounds=2)
    out = fn(spec, seed=0)
    assert len(out["y_pred"]) == len(spec.y_test)
    assert out["extra"]["rounds_run"] >= 1
    assert out["extra"]["trace"][0]["n_context"] == spec.n_train


def test_context_boosting_prune_reduces_context():
    spec = load_dataset("breast_cancer", seed=0, max_train=200, noise_rate=0.3)
    out = make_context_boosting_method("gbdt", strategy="prune",
                                       n_rounds=3, prune_ratio=0.15)(spec, 0)
    trace = out["extra"]["trace"]
    if len(trace) > 1:  # if it did not stop early, the context should shrink
        assert trace[-1]["n_context"] < trace[0]["n_context"]


def test_registry_lists():
    assert "tabpfn" in __import__("residual_lab.models", fromlist=["list_models"]).list_models()
    assert "wine" in list_datasets("classification")
