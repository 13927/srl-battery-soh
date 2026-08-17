"""Statistical tests and result aggregation."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sps


def paired_wilcoxon(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test: is method A significantly better than B?

    Returns {p_value, mean_diff, win_rate}; p=1 when all pairs are equal.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a paired test needs the two score lists to be equally long")
    diff = a - b
    if np.allclose(diff, 0):
        p = 1.0
    else:
        _, p = sps.wilcoxon(a, b)
    return {
        "p_value": float(p),
        "mean_diff": float(diff.mean()),
        "win_rate": float((diff > 0).mean()),
    }


def summarize(scores: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def compare_methods(
    results: Dict[str, List[float]], baseline: str
) -> Dict[str, Dict[str, float]]:
    """Compare every method against the baseline in a paired fashion (scores must be
    aligned by dataset and seed)."""
    if baseline not in results:
        raise ValueError(f"baseline {baseline!r} not among the results: {list(results)}")
    base = results[baseline]
    out = {}
    for name, scores in results.items():
        if name == baseline:
            continue
        entry = summarize(scores)
        entry.update(paired_wilcoxon(scores, base))
        out[name] = entry
    return out
