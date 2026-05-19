"""Train/val/test split strategies for drug response prediction.

Three protocols:
  mixed_set  - random split of (cell_line, drug) pairs
  cell_blind - held-out cell lines at test time
  drug_blind - held-out drugs at test time
"""

import numpy as np
import pandas as pd

__all__ = ['cell_blind_split', 'drug_blind_split', 'mixed_set_split']

def mixed_set_split(
    pairs: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random split of (cell_line, drug) pairs."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_test = int(len(pairs) * test_frac)
    n_val = int(len(pairs) * val_frac)
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def cell_blind_split(
    pairs: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out a fraction of cell lines entirely for test (and val)."""
    rng = np.random.default_rng(seed)
    cell_lines = pairs["depmap_id"].unique()
    cell_lines = rng.permutation(cell_lines)

    n_test_lines = max(1, int(len(cell_lines) * test_frac))
    n_val_lines = max(1, int(len(cell_lines) * val_frac))

    test_lines = set(cell_lines[:n_test_lines])
    val_lines = set(cell_lines[n_test_lines : n_test_lines + n_val_lines])

    test_idx = np.where(pairs["depmap_id"].isin(test_lines))[0]
    val_idx = np.where(pairs["depmap_id"].isin(val_lines))[0]
    train_idx = np.where(~pairs["depmap_id"].isin(test_lines | val_lines))[0]
    return train_idx, val_idx, test_idx


def drug_blind_split(
    pairs: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out a fraction of drugs entirely for test (and val)."""
    rng = np.random.default_rng(seed)
    drugs = pairs["drug_name"].unique()
    drugs = rng.permutation(drugs)

    n_test_drugs = max(1, int(len(drugs) * test_frac))
    n_val_drugs = max(1, int(len(drugs) * val_frac))

    test_drugs = set(drugs[:n_test_drugs])
    val_drugs = set(drugs[n_test_drugs : n_test_drugs + n_val_drugs])

    test_idx = np.where(pairs["drug_name"].isin(test_drugs))[0]
    val_idx = np.where(pairs["drug_name"].isin(val_drugs))[0]
    train_idx = np.where(~pairs["drug_name"].isin(test_drugs | val_drugs))[0]
    return train_idx, val_idx, test_idx
