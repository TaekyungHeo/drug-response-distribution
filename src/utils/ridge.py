"""Ridge regression helpers shared across experiment phases.

Contains safe_fit_scaler, compress_cell, compress_multi_omics — used across experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def safe_fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit StandardScaler, clamping zero-std columns to 1 to avoid NaN.

    Standard sklearn StandardScaler divides by std; if a column has zero
    variance (e.g. a MoA one-hot column with all training samples from a
    single class) it produces NaN. This wrapper replaces zero scales with 1.
    """
    sc = StandardScaler()
    sc.fit(X)
    sc.scale_ = np.where(sc.scale_ < 1e-10, 1.0, sc.scale_)  # type: ignore[operator]
    return sc


def compress_cell(
    rna: np.ndarray,
    mut: np.ndarray,
    train_cell_rows: np.ndarray,
    rna_dim: int = 550,
    mut_dim: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA-compress RNA and mutation features using training cell rows.

    PCA is fit on the training cells only (rows indexed by train_cell_rows),
    then applied to all cells. Returns compressed (float32) arrays for all
    cells; experiments index into them with cell_rows.

    Args:
        rna: shape (all_cells, n_genes), float32
        mut: shape (all_cells, n_mut_features), float32
        train_cell_rows: integer indices into rna/mut identifying training cells
        rna_dim: max PCA components for RNA
        mut_dim: max PCA components for mutations
    Returns:
        (rna_compressed, mut_compressed), each shape (all_cells, dim)
    """
    u = np.unique(train_cell_rows)

    n_rna = min(rna_dim, len(u) - 1, rna.shape[1])
    pca_rna = PCA(n_components=n_rna, random_state=42)
    pca_rna.fit(rna[u].astype(np.float64))
    rna_r = pca_rna.transform(rna.astype(np.float64)).astype(np.float32)

    n_mut = min(mut_dim, len(u) - 1, mut.shape[1])
    pca_mut = PCA(n_components=n_mut, random_state=42)
    pca_mut.fit(mut[u].astype(np.float64))
    mut_r = pca_mut.transform(mut.astype(np.float64)).astype(np.float32)

    return rna_r, mut_r


def normalize_continuous_fold(
    feat: np.ndarray,
    train_drug_idxs: np.ndarray,
) -> np.ndarray:
    """Z-score drug feature matrix using train-drug statistics.

    Args:
        feat: full drug feature matrix, shape (n_drugs, n_features)
        train_drug_idxs: row indices of training drugs in feat

    Returns:
        Normalized full matrix as float32.
    """
    sc = StandardScaler()
    sc.fit(feat[train_drug_idxs].astype(np.float64))
    sc.scale_ = np.where(sc.scale_ < 1e-10, 1.0, sc.scale_)  # type: ignore[operator]
    return sc.transform(feat.astype(np.float64)).astype(np.float32)  # type: ignore[attr-defined]


def normalize_binary_fold(
    feat: np.ndarray,
    train_drug_idxs: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Drop zero-variance binary columns and return filtered matrix.

    Removes columns where ≤1 training drug has a positive value (zero variance)
    or all training drugs have a positive value (also zero variance).

    Args:
        feat: binary drug feature matrix, shape (n_drugs, n_features)
        train_drug_idxs: row indices of training drugs in feat

    Returns:
        (filtered_feat, n_kept_columns) — feat subset to kept columns as float32.
    """
    col_sum = feat[train_drug_idxs].sum(axis=0)
    n_train = len(train_drug_idxs)
    keep = (col_sum > 1) & (col_sum < n_train)
    return feat[:, keep].astype(np.float32), int(keep.sum())


def compress_multi_omics(
    omics: dict[str, pd.DataFrame],
    modalities: list[str],
    all_cells: list[str],
    train_cells: list[str],
    pca_dims: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """PCA-compress and concatenate multiple omics modalities for all_cells.

    PCA (applied only to modalities listed in pca_dims) is fit on train_cells only,
    then applied to all_cells. Modalities absent from pca_dims are used raw.

    Args:
        omics: dict mapping modality name → DataFrame indexed by cell line
        modalities: ordered list of modality names to concatenate
        all_cells: all cell lines to build features for (train + test)
        train_cells: subset of all_cells used to fit PCA
        pca_dims: max PCA components per modality; absent modalities skip PCA.
                  Default: {"rna": 550, "mutations": 200, "cnv": 300}

    Returns:
        (cell_feat, cell_to_row) where cell_feat is shape (len(all_cells), total_dim)
        and cell_to_row maps cell id → row index in cell_feat.
    """
    if pca_dims is None:
        pca_dims = {"rna": 550, "mutations": 200, "cnv": 300}

    cell_to_row = {c: i for i, c in enumerate(all_cells)}
    train_rows = np.array([cell_to_row[c] for c in train_cells], dtype=np.int32)
    u = np.unique(train_rows)
    parts: list[np.ndarray] = []

    for mod in modalities:
        arr = omics[mod].loc[all_cells].values.astype(np.float32)
        dim = pca_dims.get(mod)
        if dim is not None:
            n_comp = min(dim, len(u) - 1, arr.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(arr[u].astype(np.float64))
            feat = pca.transform(arr.astype(np.float64)).astype(np.float32)
        else:
            feat = arr
        parts.append(feat)

    return np.concatenate(parts, axis=1), cell_to_row


__all__ = [
    "compress_cell",
    "compress_multi_omics",
    "normalize_binary_fold",
    "normalize_continuous_fold",
    "safe_fit_scaler",
]
