"""Experiment-matrix runner: dataset x method x noise x seed, written to
JSONL/CSV."""

from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from residual_lab.data import DatasetSpec, load_dataset
from residual_lab.metrics import PRIMARY_METRIC, evaluate

# method signature: (spec, seed) -> {"y_pred": ..., "y_proba": ... or None,
#                                    "extra": dict}
MethodFn = Callable[[DatasetSpec, int], Dict[str, Any]]


@dataclass
class RunRecord:
    dataset: str
    task: str
    method: str
    seed: int
    noise_rate: float
    n_train: int
    metrics: Dict[str, float]
    runtime_s: float
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def primary_score(self) -> Optional[float]:
        return self.metrics.get(PRIMARY_METRIC[self.task])


def sklearn_method(model_name: str) -> MethodFn:
    """Wrap a registered model into a method the runner can execute."""
    from residual_lab.models import make_model

    def fn(spec: DatasetSpec, seed: int) -> Dict[str, Any]:
        model = make_model(model_name, seed=seed)
        model.fit(spec.X_train, spec.y_train)
        y_pred = model.predict(spec.X_test)
        y_proba = None
        if spec.task == "classification" and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(spec.X_test)
        return {"y_pred": y_pred, "y_proba": y_proba, "extra": {}}

    return fn


class ExperimentRunner:
    """Run the matrix and append results incrementally (one JSONL line at a time,
    so an interrupted run can resume)."""

    def __init__(self, out_dir: str = "results", exp_name: str = "exp"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / f"{exp_name}.jsonl"
        self.records: List[RunRecord] = []
        self._done = self._load_done_keys()

    def _load_done_keys(self) -> set:
        done = set()
        if self.jsonl_path.exists():
            for line in self.jsonl_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    if r.get("error") is None:
                        done.add((r["dataset"], r["method"], r["seed"], r["noise_rate"]))
                    self.records.append(RunRecord(**r))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
        return done

    def run_matrix(
        self,
        datasets: Sequence[str],
        methods: Dict[str, MethodFn],
        seeds: Sequence[int] = (0, 1, 2),
        noise_rates: Sequence[float] = (0.0,),
        max_train: int = 500,
        verbose: bool = True,
    ) -> List[RunRecord]:
        for ds_name in datasets:
            for noise in noise_rates:
                for seed in seeds:
                    spec = load_dataset(ds_name, seed=seed, max_train=max_train, noise_rate=noise)
                    for m_name, m_fn in methods.items():
                        key = (ds_name, m_name, seed, noise)
                        if key in self._done:
                            continue
                        rec = self._run_one(spec, m_name, m_fn, seed)
                        self._append(rec)
                        if verbose:
                            score = rec.primary_score
                            msg = f"{score:.4f}" if score is not None else f"ERROR: {rec.error}"
                            print(f"[{ds_name} noise={noise} seed={seed}] {m_name}: "
                                  f"{msg} ({rec.runtime_s:.1f}s)")
        return self.records

    def _run_one(self, spec: DatasetSpec, m_name: str, m_fn: MethodFn, seed: int) -> RunRecord:
        t0 = time.time()
        try:
            out = m_fn(spec, seed)
            metrics = evaluate(spec.task, spec.y_test, out["y_pred"], out.get("y_proba"))
            return RunRecord(
                dataset=spec.name, task=spec.task, method=m_name, seed=seed,
                noise_rate=spec.noise_rate, n_train=spec.n_train,
                metrics=metrics, runtime_s=time.time() - t0,
                extra=out.get("extra", {}),
            )
        except Exception:
            return RunRecord(
                dataset=spec.name, task=spec.task, method=m_name, seed=seed,
                noise_rate=spec.noise_rate, n_train=spec.n_train,
                metrics={}, runtime_s=time.time() - t0,
                error=traceback.format_exc(limit=3),
            )

    def _append(self, rec: RunRecord) -> None:
        self.records.append(rec)
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(asdict(rec), default=_json_default) + "\n")

    # ---- aggregation ----

    def scores_by_method(self, metric: Optional[str] = None) -> Dict[str, List[float]]:
        """Per-method score series aligned by (dataset, noise, seed), for
        stats.compare_methods."""
        ok = [r for r in self.records if r.error is None]
        keys = sorted({(r.dataset, r.noise_rate, r.seed) for r in ok})
        methods = sorted({r.method for r in ok})
        out: Dict[str, List[float]] = {m: [] for m in methods}
        for key in keys:
            row = {r.method: r for r in ok
                   if (r.dataset, r.noise_rate, r.seed) == key}
            if len(row) < len(methods):
                continue  # keep only configurations finished by every method, so
                          # the comparison stays paired
            for m in methods:
                r = row[m]
                out[m].append(r.metrics[metric or PRIMARY_METRIC[r.task]])
        return out

    def to_csv(self, path: Optional[str] = None) -> Path:
        path = Path(path) if path else self.jsonl_path.with_suffix(".csv")
        ok = [r for r in self.records if r.error is None]
        metric_names = sorted({k for r in ok for k in r.metrics})
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "task", "method", "seed", "noise_rate",
                        "n_train", "runtime_s"] + metric_names)
            for r in ok:
                w.writerow([r.dataset, r.task, r.method, r.seed, r.noise_rate,
                            r.n_train, f"{r.runtime_s:.2f}"]
                           + [r.metrics.get(m, "") for m in metric_names])
        return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"cannot serialise {type(o)}")
