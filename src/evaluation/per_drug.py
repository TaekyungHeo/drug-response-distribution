"""Per-drug evaluation utilities shared across all experiment phases.

These functions are duplicated across ~14 experiment scripts.
Canonical versions live here; experiments should import from here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def per_drug_r(
    preds: np.ndarray,
    targets: np.ndarray,
    drug_names: np.ndarray,
    min_cells: int = 5,
    metric: str = "pearson",
) -> dict[str, float]:
    """Compute per-drug Pearson (or Spearman) r, averaging across cell lines.

    Args:
        preds: predicted values, shape (N,)
        targets: true values, shape (N,)
        drug_names: drug name per sample, shape (N,)
        min_cells: minimum cell lines per drug to include
        metric: "pearson" or "spearman"
    Returns:
        Dict mapping drug name → correlation coefficient
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    drug_names = np.asarray(drug_names)

    rs: dict[str, float] = {}
    for d in np.unique(drug_names):
        m = drug_names == d
        if m.sum() < min_cells:
            continue
        p, t = preds[m], targets[m]
        if t.std() < 1e-8 or p.std() < 1e-8:
            continue
        if metric == "pearson":
            rs[d] = float(pearsonr(p, t)[0])
        else:
            rs[d] = float(spearmanr(p, t)[0])
    return rs


def mean_per_drug_r(
    preds: np.ndarray,
    targets: np.ndarray,
    drug_names: np.ndarray,
    min_cells: int = 5,
    metric: str = "pearson",
) -> float:
    """Mean per-drug r across all drugs with sufficient cell coverage."""
    rs = per_drug_r(preds, targets, drug_names, min_cells=min_cells, metric=metric)
    if not rs:
        return float("nan")
    return float(np.mean(list(rs.values())))


def per_moa_r(
    preds: np.ndarray,
    targets: np.ndarray,
    drug_names: np.ndarray,
    drug_moa: dict[str, str],
    focus_moa: list | None = None,
    min_cells: int = 5,
) -> dict[str, float]:
    """Mean per-drug r broken down by MoA class.

    Args:
        drug_moa: dict mapping drug name → MoA label
        focus_moa: if given, only include these MoA labels
    Returns:
        Dict mapping MoA label → mean per-drug r
    """
    rs = per_drug_r(preds, targets, drug_names, min_cells=min_cells)
    moa_groups: dict[str, list] = {}
    for drug, r in rs.items():
        moa = drug_moa.get(drug, "Unknown")
        if focus_moa is not None and moa not in focus_moa:
            continue
        moa_groups.setdefault(moa, []).append(r)
    return {moa: float(np.mean(vals)) for moa, vals in moa_groups.items() if vals}


def per_cell_r(
    preds: np.ndarray,
    targets: np.ndarray,
    cell_ids: np.ndarray,
    min_drugs: int = 5,
    metric: str = "pearson",
) -> dict[str, float]:
    """Compute per-cell Pearson (or Spearman) r, averaging across drugs.

    Args:
        preds: predicted values, shape (N,)
        targets: true values, shape (N,)
        cell_ids: cell identifier per sample, shape (N,)
        min_drugs: minimum drugs per cell to include
        metric: "pearson" or "spearman"
    Returns:
        Dict mapping cell_id → correlation coefficient
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    cell_ids = np.asarray(cell_ids)

    rs: dict[str, float] = {}
    for c in np.unique(cell_ids):
        m = cell_ids == c
        if m.sum() < min_drugs:
            continue
        p, t = preds[m], targets[m]
        if t.std() < 1e-8 or p.std() < 1e-8:
            continue
        if metric == "pearson":
            rs[str(c)] = float(pearsonr(p, t)[0])  # type: ignore[arg-type]
        else:
            rs[str(c)] = float(spearmanr(p, t)[0])  # type: ignore[arg-type]
    return rs


def mean_per_cell_r(
    preds: np.ndarray,
    targets: np.ndarray,
    cell_ids: np.ndarray,
    min_drugs: int = 5,
    metric: str = "pearson",
) -> float:
    """Mean per-cell r across all cell lines with sufficient drug coverage."""
    rs = per_cell_r(preds, targets, cell_ids, min_drugs=min_drugs, metric=metric)
    if not rs:
        return float("nan")
    return float(np.mean(list(rs.values())))


__all__ = ["mean_per_cell_r", "mean_per_drug_r", "per_cell_r", "per_drug_r", "per_moa_r"]
