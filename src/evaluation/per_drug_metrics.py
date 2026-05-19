"""Single-drug metric functions and bootstrap CI utility.

Each function receives y_true and y_pred for one drug (one fold) and returns
a scalar. Use compute_all() to get all five candidate metrics at once.

Design notes:
- r2_drug_mean uses per-drug mean as SS_tot baseline (NOT sklearn r2_score, which
  uses the test-set global mean — wrong for per-drug evaluation).
- ndcg_at_5 shifts y_true to [0, ...] because sklearn.metrics.ndcg_score
  requires non-negative relevance scores; the shift preserves rank order.
- On a constant-within-drug predictor (Predictor 2A), Pearson/Spearman/Kendall
  return 0.0 by convention; NDCG returns a near-maximal value (sklearn treats
  all-tied predictions as equally valid orderings). Callers should record NDCG
  on 2A but not treat it as a pass/fail sanity check.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import ndcg_score


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-6:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-6:
        return 0.0
    return float(spearmanr(y_true, y_pred)[0])


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-6:
        return 0.0
    return float(kendalltau(y_true, y_pred)[0])


def ndcg_at_5(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NDCG@5 via sklearn. y_true is shifted to ≥ 0 to satisfy sklearn's constraint."""
    if len(y_true) < 5:
        return float("nan")
    shifted = y_true - y_true.min()
    return float(ndcg_score(shifted.reshape(1, -1), y_pred.reshape(1, -1), k=5))


def r2_drug_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² with per-drug mean as baseline.

    SS_tot = sum((y_true - mean(y_true))^2). Returns 0.0 when y_true is constant
    (undefined case). This is NOT sklearn r2_score, which uses the global test-set
    mean and would give incorrect values in per-drug evaluation.
    """
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All five candidate metrics for a single (drug, fold) pair."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "r_p": pearson_r(y_true, y_pred),
        "r_s": spearman_r(y_true, y_pred),
        "tau": kendall_tau(y_true, y_pred),
        "ndcg5": ndcg_at_5(y_true, y_pred),
        "r2": r2_drug_mean(y_true, y_pred),
    }


def bootstrap_ci_width(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 200,
    seed: int = 0,
) -> float:
    """95% bootstrap CI width for a per-drug metric function.

    Resamples with replacement n_boot times. Returns NaN if fewer than 10
    non-NaN samples result (too few cells to estimate CI reliably).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = metric_fn(y_true[idx], y_pred[idx])
        if not np.isnan(val):
            samples.append(val)
    if len(samples) < 10:
        return float("nan")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(hi - lo)


__all__ = [
    "bootstrap_ci_width",
    "compute_all",
    "kendall_tau",
    "ndcg_at_5",
    "pearson_r",
    "r2_drug_mean",
    "spearman_r",
]
