"""PASO pre-generated drug-blind fold loader.

Every drug-blind experiment uses the same PASO 5- or 10-fold splits.
This module centralises the loading logic (duplicated in every experiment).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_PASO_DIR = (
    Path(__file__).parents[2] / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"
)


def load_paso_folds(
    n_folds: int = 5,
    paso_dir: Path | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Load PASO pre-generated drug-blind CV splits.

    Args:
        n_folds: number of folds to load (5 or 10)
        paso_dir: path to fold directory; defaults to standard location

    Returns:
        List of (train_df, test_df) DataFrames with columns
        ['drug', 'cell_line', 'IC50']
    """
    d = paso_dir or _DEFAULT_PASO_DIR
    return [
        (
            pd.read_csv(d / f"DrugBlind_train_Fold{i}.csv"),
            pd.read_csv(d / f"DrugBlind_test_Fold{i}.csv"),
        )
        for i in range(n_folds)
    ]


def build_pair_index(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    name_to_depmap: dict,
    rna_index: pd.Index,
    mut_index: pd.Index,
) -> tuple[pd.DataFrame, dict]:
    """Build a deduplicated pair dataframe and lookup index from PASO folds.

    Args:
        folds: output of load_paso_folds
        name_to_depmap: dict mapping stripped cell line name (upper) → DepMap ID
        rna_index: pandas Index of cell lines with RNA data
        mut_index: pandas Index of cell lines with mutation data

    Returns:
        (full_df, key_to_idx) where full_df has columns
        [depmap_id, drug_name, ic50] and key_to_idx maps
        (depmap_id, drug_name) → row index in full_df
    """
    all_pairs = pd.concat([pd.concat([tr, te]) for tr, te in folds]).drop_duplicates(
        subset=["drug", "cell_line"]
    )

    valid_rows = []
    for _, row in all_pairs.iterrows():
        dep = name_to_depmap.get(str(row["cell_line"]).upper())
        drug = row["drug"]
        if dep and dep in rna_index and dep in mut_index:
            valid_rows.append({"depmap_id": dep, "drug_name": drug, "ic50": float(str(row["IC50"]))})

    full_df = pd.DataFrame(valid_rows)
    key_to_idx = {(row["depmap_id"], row["drug_name"]): i for i, row in full_df.iterrows()}
    return full_df, key_to_idx


def map_fold_indices(
    df: pd.DataFrame,
    key_to_idx: dict,
    name_to_depmap: dict,
) -> np.ndarray:
    """Map a fold DataFrame (drug, cell_line, IC50) to row indices in full_df."""
    idx = []
    for _, row in df.iterrows():
        dep = name_to_depmap.get(str(row["cell_line"]).upper())
        drug = row["drug"]
        if dep and (dep, drug) in key_to_idx:
            idx.append(key_to_idx[(dep, drug)])
    return np.array(idx, dtype=np.int64)


def load_cell_line_index(data_dir: Path) -> dict[str, str]:
    """Build name_to_depmap mapping from cell_line_index.parquet.

    Returns:
        Dict mapping stripped cell line name (upper-cased) → DepMap ID string.
    """
    cl_idx = pd.read_parquet(data_dir / "cell_line_index.parquet")
    name_to_depmap: dict[str, str] = {}
    for depmap_id, row in cl_idx.iterrows():
        name_to_depmap[str(row["stripped_name"]).upper()] = str(depmap_id)
    return name_to_depmap


def load_paso_pairs(
    paso_folds_dir: Path,
    name_to_depmap: dict[str, str],
    available_cells: set,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PASO fold k and map cell_line names to DepMap IDs.

    Args:
        paso_folds_dir: path to the directory containing DrugBlind_*_Fold*.csv files
        name_to_depmap: mapping from stripped_name (upper) → depmap_id
        available_cells: set of depmap_ids with usable omics features
        k: fold index (0-based)

    Returns:
        (train_df, test_df) each with columns: depmap_id, drug_name, ln_ic50.
        Only rows where the cell line has omics data are included.
    """
    def _map(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            ccl = str(row["cell_line"]).upper()
            depmap: str | None = name_to_depmap.get(ccl)
            drug = row["drug"]
            if depmap and depmap in available_cells:
                rows.append({
                    "depmap_id": depmap,
                    "drug_name": drug,
                    "ln_ic50": float(str(row["IC50"])),  # type: ignore[arg-type]
                })
        return pd.DataFrame(rows)

    train_raw = pd.read_csv(paso_folds_dir / f"DrugBlind_train_Fold{k}.csv")
    test_raw = pd.read_csv(paso_folds_dir / f"DrugBlind_test_Fold{k}.csv")
    return _map(train_raw), _map(test_raw)


def split_drug_blind_val(
    drug_idxs_arr: np.ndarray,
    full_train_idx: np.ndarray,
    fold_i: int,
    val_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a drug-blind val set out of a training index array.

    Randomly assigns val_frac of train drugs to a validation split.
    The same drug never appears in both train and val.

    Args:
        drug_idxs_arr: per-pair drug index array (length = all pairs)
        full_train_idx: indices into drug_idxs_arr for the current fold's train pairs
        fold_i: fold index used to seed the RNG (deterministic per fold)
        val_frac: fraction of train drugs to hold out

    Returns:
        (train_idx, val_idx) — disjoint subsets of full_train_idx
    """
    train_drug_indices = np.unique(drug_idxs_arr[full_train_idx])
    rng = np.random.default_rng(42 + fold_i)
    shuffled = rng.permutation(len(train_drug_indices))
    n_val = max(1, int(len(train_drug_indices) * val_frac))
    val_drug_set = set(train_drug_indices[shuffled[:n_val]].tolist())
    val_mask = np.isin(drug_idxs_arr[full_train_idx], list(val_drug_set))
    return full_train_idx[~val_mask], full_train_idx[val_mask]


__all__ = [
    "build_pair_index",
    "load_cell_line_index",
    "load_paso_folds",
    "load_paso_pairs",
    "map_fold_indices",
    "split_drug_blind_val",
]
