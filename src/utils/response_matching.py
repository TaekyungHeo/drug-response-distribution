"""Response profile matching utilities — Solution 2 of the three-strategy framework.

Core algorithms for K-shot drug response prediction via collaborative filtering:
- build_response_matrix: construct drug×cell IC50 matrix from a DataFrame
- response_match: predict unseen cell responses given K observations

These functions were duplicated in ~6 experiment scripts. Canonical versions live here.

Paper reference: §Results "Response profile matching improves within-drug ranking"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "build_response_matrix",
    "response_match",
    "select_cells_diverse",
    "select_cells_maxvar",
    "select_cells_midresp",
]

# Default number of nearest training drugs used for transfer
_DEFAULT_TOP_N = 5


def build_response_matrix(
    dr: pd.DataFrame,
    drugs: list[str],
    cells: list[str],
) -> np.ndarray:
    """Build a (n_drugs, n_cells) response matrix from a long-format DataFrame.

    Args:
        dr: DataFrame with columns [drug_name, depmap_id, ln_ic50].
        drugs: Ordered drug list — row order of the output matrix.
        cells: Ordered cell list — column order of the output matrix.
    Returns:
        float32 matrix with NaN for missing entries, shape (len(drugs), len(cells)).
    """

    drug_idx = {d: i for i, d in enumerate(drugs)}
    cell_idx = {c: i for i, c in enumerate(cells)}
    mat = np.full((len(drugs), len(cells)), np.nan, dtype=np.float32)
    for _, row in dr.iterrows():
        d, c = row["drug_name"], row["depmap_id"]
        if d in drug_idx and c in cell_idx:
            mat[drug_idx[d], cell_idx[c]] = row["ln_ic50"]
    return mat


def response_match(
    obs_cells: np.ndarray,
    obs_vals: np.ndarray,
    train_mat: np.ndarray,
    pred_cells: np.ndarray,
    K: int,
    top_n: int = _DEFAULT_TOP_N,
) -> np.ndarray:
    """Predict IC50 for pred_cells using K observations of the test drug.

    Implements the collaborative filtering core:
    - K=0: return training mean for each pred cell
    - K=1: match by nearest IC50 value (potency calibration)
    - K≥2: match by Pearson correlation of the K-observation profile

    Args:
        obs_cells: Column indices of observed cell lines (shape K,).
        obs_vals: Observed IC50 values for those cells (shape K,).
        train_mat: (n_train_drugs, n_cells) matrix of training responses.
        pred_cells: Column indices of cells to predict.
        K: Number of observations. Must equal len(obs_cells).
        top_n: Number of nearest training drugs used for prediction.
    Returns:
        Predicted IC50 values for pred_cells, shape (len(pred_cells),).
    """
    if K == 0:
        return np.nanmean(train_mat[:, pred_cells], axis=0)

    if K == 1:
        col = train_mat[:, obs_cells[0]]
        valid = ~np.isnan(col)
        if valid.sum() < top_n:
            return np.nanmean(train_mat[:, pred_cells], axis=0)
        dists = np.where(valid, np.abs(col - obs_vals[0]), np.inf)
        top = np.argsort(dists)[:top_n]
        w = np.ones(top_n, dtype=np.float32)
    else:
        corrs = np.full(len(train_mat), np.nan)
        for i in range(len(train_mat)):
            ref = train_mat[i, obs_cells]
            ok = ~np.isnan(ref)
            if ok.sum() < max(3, K // 2):
                continue
            if obs_vals[ok].std() < 1e-8 or ref[ok].std() < 1e-8:
                continue
            corrs[i] = float(pearsonr(obs_vals[ok], ref[ok])[0])
        valid_corrs = ~np.isnan(corrs)
        if valid_corrs.sum() < top_n:
            return np.nanmean(train_mat[:, pred_cells], axis=0)
        valid_idx = np.where(valid_corrs)[0]
        top = valid_idx[np.argsort(corrs[valid_idx])[-top_n:]][::-1]
        w = np.maximum(corrs[top], 0).astype(np.float32)

    if w.sum() < 1e-10:
        w = np.ones(top_n, dtype=np.float32)
    w = w / w.sum()

    preds = np.full(len(pred_cells), np.nan, dtype=np.float32)
    for j, pc in enumerate(pred_cells):
        col = train_mat[top, pc]
        ok = ~np.isnan(col)
        if ok.sum() == 0:
            preds[j] = float(np.nanmean(train_mat[:, pc]))
        else:
            w_ok = w[ok]
            w_sum = w_ok.sum()
            if w_sum < 1e-10:
                w_ok = np.ones(ok.sum(), dtype=np.float32)
                w_sum = w_ok.sum()
            preds[j] = float(np.dot(w_ok / w_sum, col[ok]))
    return preds


# ── Active cell selection strategies ─────────────────────────────────────────


def select_cells_maxvar(
    K: int,
    valid_cells: np.ndarray,
    train_mat: np.ndarray,
) -> np.ndarray:
    """Select K cells with highest response variance across training drugs.

    Best at small K (anchors potency class). See §Results §Active learning.
    """
    var = np.nanvar(train_mat[:, valid_cells], axis=0)
    return valid_cells[np.argsort(var)[::-1][:K]]


def select_cells_midresp(
    K: int,
    valid_cells: np.ndarray,
    train_mat: np.ndarray,
) -> np.ndarray:
    """Select K cells closest to the median drug response (representative cells).

    Best at larger K (K≥10). See §Results §Active learning.
    """
    medians = np.nanmedian(train_mat[:, valid_cells], axis=0)
    grand_median = float(np.nanmedian(train_mat))
    return valid_cells[np.argsort(np.abs(medians - grand_median))[:K]]


def select_cells_diverse(
    K: int,
    valid_cells: np.ndarray,
    train_mat: np.ndarray,
) -> np.ndarray:
    """Greedy farthest-point selection in drug-response space.

    Covers response space maximally. Falls back to MaxVar at K=1.
    """
    if K == 1:
        return select_cells_maxvar(1, valid_cells, train_mat)

    profiles = train_mat[:, valid_cells].T  # (n_cells, n_drugs)
    col_means = np.nanmean(profiles, axis=0)
    profiles = np.where(np.isnan(profiles), col_means[None, :], profiles)

    first = int(np.argmax(np.nanvar(train_mat[:, valid_cells], axis=0)))
    selected = [first]
    for _ in range(K - 1):
        min_dists = np.full(len(valid_cells), np.inf)
        for s in selected:
            d = np.sum((profiles - profiles[s][None, :]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d)
        selected.append(int(np.argmax(min_dists)))
    return valid_cells[np.array(selected)]
