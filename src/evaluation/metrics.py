"""Evaluation metrics for drug response prediction."""

import numpy as np
from scipy import stats


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    r, _ = stats.spearmanr(y_true, y_pred)
    return float(r)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_r": spearman_r(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "n": len(y_true),
    }


def evaluate_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drug_names: np.ndarray,
    cell_ids: np.ndarray,
) -> dict[str, object]:
    """Evaluate global r, per-drug r, and per-cell r in one call.

    Returns a dict suitable for JSON serialisation.
    """
    from src.evaluation.per_drug import mean_per_cell_r, mean_per_drug_r

    return {
        "global_r": pearson_r(y_true, y_pred),
        "per_drug_r": mean_per_drug_r(y_pred, y_true, drug_names),
        "per_cell_r": mean_per_cell_r(y_pred, y_true, cell_ids),
        "n": len(y_true),
    }


__all__ = ["evaluate", "evaluate_full", "pearson_r", "rmse", "spearman_r"]
