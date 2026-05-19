"""Utilities for loading and preprocessing omics features.

Shared across experiment scripts that use cell-only omics as input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

MODALITY_FILES: dict[str, str] = {
    "rna": "rna.parquet",
    "mutations": "mutations.parquet",
    "cnv": "cnv.parquet",
    "metabolomics": "metabolomics.parquet",
    "rppa": "rppa.parquet",
}


def load_omics(
    modalities: list[str],
    cell_lines: list[str],
    processed_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load and concatenate omics features for the given cell lines.

    Args:
        modalities: list of modality names, e.g. ["rna", "mutations"]
        cell_lines: DepMap IDs to include
        processed_dir: path to processed data directory; defaults to repo default

    Returns:
        (matrix, cell_order) where matrix has shape (n_cells, total_features)
        and cell_order is the sorted list of DepMap IDs that appear in all modalities.
    """
    base = processed_dir or PROCESSED_DIR
    cell_set = set(cell_lines)

    dfs: list[pd.DataFrame] = []
    for mod in modalities:
        fname = MODALITY_FILES.get(mod, f"{mod}.parquet")
        df = pd.read_parquet(base / fname)
        df = df.loc[df.index.isin(cell_set)]
        dfs.append(df)

    # Intersect: keep only cells present in ALL modalities
    common = sorted(set(dfs[0].index))
    for df in dfs[1:]:
        common = sorted(set(common) & set(df.index))

    mat = np.concatenate(
        [df.loc[common].to_numpy(dtype=np.float32) for df in dfs],
        axis=1,
    )
    return mat, common


def build_pair_features(
    pairs: pd.DataFrame,
    cell_mat: np.ndarray,
    cell_order: list[str],
) -> np.ndarray:
    """Map (cell, drug) pairs to per-pair feature rows.

    Args:
        pairs: DataFrame with 'depmap_id' column
        cell_mat: (n_cells, n_features) matrix aligned to cell_order
        cell_order: sorted list of DepMap IDs (same ordering as cell_mat rows)

    Returns:
        (n_pairs, n_features) float32 array
    """
    c2r: dict[str, int] = {c: i for i, c in enumerate(cell_order)}
    row_idx = np.array([c2r[c] for c in pairs["depmap_id"]], dtype=np.intp)
    return cell_mat[row_idx]


def z_score_normalize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise using train-set statistics; clamp zero-std columns.

    Args:
        X_train, X_val, X_test: float arrays, all with the same number of columns

    Returns:
        (X_train_norm, X_val_norm, X_test_norm) as float32 arrays
    """
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std < 1e-9] = 1.0

    def _norm(X: np.ndarray) -> np.ndarray:
        # copy=False: if X is already float32, returns X itself (no extra allocation).
        # pair_X[idx] is always a copy from fancy indexing, so in-place ops are safe.
        out = X.astype(np.float32, copy=False)
        out -= mean
        out /= std
        return out

    return _norm(X_train), _norm(X_val), _norm(X_test)


__all__ = ["build_pair_features", "load_omics", "z_score_normalize"]
