"""Shared utilities for the 05_solutions experiment series.

Functions for MoA annotation loading, weighted Ridge regression,
K-shot response matching, and profile concordance analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_DEFAULT_MOA_PATH = Path("external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv")


def load_moa_annotations(moa_file: str | Path | None = None) -> dict[str, str]:
    """Load drug name -> Target Pathway mapping from the PASO annotation CSV.

    Args:
        moa_file: Path to the CSV. Defaults to the PASO annotation file
                  relative to the project root.

    Returns:
        Dict mapping drug name (str) to pathway label (str).
    """
    if moa_file is None:
        moa_file = _DEFAULT_MOA_PATH
    df = pd.read_csv(moa_file)
    # Columns: unnamed index, "Drug name", "Drug ID", "Drug target", "Target Pathway"
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        name = row["Drug name"]
        pathway = row["Target Pathway"]
        if pd.notna(name) and pd.notna(pathway):
            out[str(name)] = str(pathway)
    return out


def group_drugs_by_moa(
    drug_names: list[str] | np.ndarray,
    moa: dict[str, str],
) -> dict[str, list[str]]:
    """Group drug names by MoA class, dropping drugs absent from *moa*.

    Args:
        drug_names: Iterable of drug name strings.
        moa: Dict mapping drug name -> MoA label (from load_moa_annotations).

    Returns:
        Dict mapping MoA label -> list of drug names in that class.
    """
    groups: dict[str, list[str]] = {}
    for d in drug_names:
        label = moa.get(str(d))
        if label is not None:
            groups.setdefault(label, []).append(str(d))
    return groups


def fit_weighted_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray | None = None,
    alpha: float = 1.0,
) -> Ridge:
    """Fit a Ridge regression with optional per-sample weights.

    Args:
        X_train: Training features, shape (n_samples, n_features).
        y_train: Training targets, shape (n_samples,).
        sample_weights: Per-sample weights, shape (n_samples,). None = uniform.
        alpha: Ridge regularization strength.

    Returns:
        Fitted sklearn Ridge model.
    """
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def response_match_predict(
    train_response_matrix: np.ndarray,
    test_observed: np.ndarray,
    anchor_cell_idx: np.ndarray,
    cell_mean: np.ndarray,
    blend_weight: float = 0.5,
    n_neighbors: int = 5,
) -> np.ndarray:
    """K-shot response matching prediction.

    If K=0 (test_observed is empty), returns cell_mean directly.
    Otherwise, finds the n_neighbors most similar training drugs based on
    anchor cell responses, weighted-averages their full profiles, and blends
    with cell_mean.

    Args:
        train_response_matrix: (n_train_drugs, n_cells) response matrix. May contain NaN.
        test_observed: Observed IC50 values at anchor cells for the test drug, shape (K,).
        anchor_cell_idx: Column indices of anchor cells, shape (K,).
        cell_mean: Per-cell mean response from training data, shape (n_cells,).
        blend_weight: Weight on neighbor prediction vs cell_mean (1.0 = all neighbors).
        n_neighbors: Number of nearest training drugs to use.

    Returns:
        Predicted response profile, shape (n_cells,).
    """
    cell_mean = np.asarray(cell_mean, dtype=np.float64)

    if len(test_observed) == 0:
        return cell_mean.copy()

    test_observed = np.asarray(test_observed, dtype=np.float64)
    anchor_cell_idx = np.asarray(anchor_cell_idx, dtype=int)

    # Compute similarity to each training drug on anchor cells
    # Vectorized path for rows without NaN; inline-numpy fallback for rows with NaN.
    n_train = train_response_matrix.shape[0]
    similarities = np.full(n_train, -np.inf)

    anchor_slice = train_response_matrix[:, anchor_cell_idx].astype(np.float64)  # (n_train, K)
    t_ok = ~np.isnan(test_observed)
    if t_ok.sum() < 2:
        return cell_mean.copy()

    A = anchor_slice[:, t_ok]   # (n_train, n_ok)
    t = test_observed[t_ok]     # (n_ok,)
    n_ok = int(t_ok.sum())
    t_mean = t.mean()
    t_std = t.std()

    if t_std >= 1e-8:
        row_has_nan = np.isnan(A).any(axis=1)  # (n_train,)
        clean = np.where(~row_has_nan)[0]

        # Vectorized Pearson for clean rows
        if len(clean) > 0:
            Ac = A[clean]                                # (n_clean, n_ok)
            Ac_mean = Ac.mean(axis=1, keepdims=True)     # (n_clean, 1)
            Ac_std = Ac.std(axis=1)                      # (n_clean,)
            valid = Ac_std >= 1e-8
            if valid.any():
                num = ((Ac[valid] - Ac_mean[valid]) * (t - t_mean)).sum(axis=1)
                similarities[clean[valid]] = num / (n_ok * Ac_std[valid] * t_std)

        # Inline-numpy fallback for rows with NaN (avoids scipy overhead)
        for i in np.where(row_has_nan)[0]:
            row = A[i]
            ok = ~np.isnan(row)
            if ok.sum() < 2:
                continue
            t_ok_vals = t[ok]
            ref_ok = row[ok]
            if t_ok_vals.std() < 1e-8 or ref_ok.std() < 1e-8:
                continue
            t_c = t_ok_vals - t_ok_vals.mean()
            r_c = ref_ok - ref_ok.mean()
            denom = np.sqrt((t_c**2).sum() * (r_c**2).sum())
            if denom > 1e-10:
                similarities[i] = float((t_c * r_c).sum() / denom)

    valid_mask = similarities > -np.inf
    if valid_mask.sum() < n_neighbors:
        return cell_mean.copy()

    valid_idx = np.where(valid_mask)[0]
    top_idx = valid_idx[np.argsort(similarities[valid_idx])[-n_neighbors:]]

    # Weight by positive correlation
    weights = np.maximum(similarities[top_idx], 0.0)
    if weights.sum() < 1e-10:
        weights = np.ones(len(top_idx))
    weights = weights / weights.sum()

    # Vectorized neighbor profile prediction — renormalize per cell for NaN entries
    nb_slice = train_response_matrix[top_idx, :].astype(np.float64)  # (n_neighbors, n_cells)
    nan_nb = np.isnan(nb_slice)
    w_mat = weights[:, np.newaxis] * ~nan_nb        # zero weight on NaN cells
    w_sum = w_mat.sum(axis=0)                        # (n_cells,)
    filled = np.where(nan_nb, 0.0, nb_slice)
    has_data = w_sum > 1e-10
    neighbor_pred = np.where(has_data, (w_mat * filled).sum(axis=0) / np.where(has_data, w_sum, 1.0), cell_mean)

    return blend_weight * neighbor_pred + (1.0 - blend_weight) * cell_mean


def pairwise_profile_concordance(
    response_matrix: np.ndarray,
    drug_names: list[str] | np.ndarray,
    drug_groups: dict[str, list[str]],
    min_shared_cells: int = 20,
) -> dict[str, dict]:
    """Compute within-group pairwise Pearson r between drug response profiles.

    For each MoA group with >= 2 drugs, compute all pairwise correlations
    between response profiles (columns = cell lines), using only cell lines
    where both drugs have non-NaN values.

    Args:
        response_matrix: (n_drugs, n_cells) matrix, may contain NaN.
        drug_names: Drug names corresponding to rows of response_matrix.
        drug_groups: Dict mapping group label -> list of drug names.
        min_shared_cells: Minimum number of shared non-NaN cells for a pair.

    Returns:
        Dict mapping group label -> {mean_r, std_r, n_pairs, n_drugs}.
        Groups with < 2 drugs or no valid pairs are excluded.
    """
    drug_names = list(drug_names)
    name_to_idx = {d: i for i, d in enumerate(drug_names)}

    results: dict[str, dict] = {}
    for group, members in drug_groups.items():
        # Filter to members present in the matrix
        idxs = [name_to_idx[d] for d in members if d in name_to_idx]
        if len(idxs) < 2:
            continue

        pair_rs: list[float] = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                prof_a = response_matrix[idxs[a]].astype(np.float64)
                prof_b = response_matrix[idxs[b]].astype(np.float64)
                ok = ~np.isnan(prof_a) & ~np.isnan(prof_b)
                if ok.sum() < min_shared_cells:
                    continue
                pa, pb = prof_a[ok], prof_b[ok]
                if pa.std() < 1e-8 or pb.std() < 1e-8:
                    continue
                pa_c, pb_c = pa - pa.mean(), pb - pb.mean()
                denom = np.sqrt((pa_c**2).sum() * (pb_c**2).sum())
                if denom > 1e-10:
                    pair_rs.append(float((pa_c * pb_c).sum() / denom))

        if pair_rs:
            results[group] = {
                "mean_r": float(np.mean(pair_rs)),
                "std_r": float(np.std(pair_rs)),
                "n_pairs": len(pair_rs),
                "n_drugs": len(idxs),
            }

    return results


__all__ = [
    "fit_weighted_ridge",
    "group_drugs_by_moa",
    "load_moa_annotations",
    "pairwise_profile_concordance",
    "response_match_predict",
]
