"""Experiment-local data loading for 01_global_vs_perdrug.

Provides a unified DataBundle that all four job scripts consume.
Shared utilities (omics_utils, splits, paso_folds) stay in the repo src/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[4]
DATA_DIR = ROOT / "data" / "processed"

OMICS = ["rna", "mutations"]


@dataclass
class DataBundle:
    """All arrays needed to slice train/val/test folds."""
    full_df: pd.DataFrame          # (n_pairs,) [depmap_id, drug_name, ic50]
    concat_np: np.ndarray          # (n_cells, rna+mut)
    cell_rows: np.ndarray          # (n_pairs,) int32 → row in concat_np
    drug_idxs: np.ndarray          # (n_pairs,) int32 → row in fp_matrix
    targets: np.ndarray            # (n_pairs,) float32 ln_ic50
    drug_names: np.ndarray         # (n_pairs,) str
    cell_ids: np.ndarray           # (n_pairs,) str depmap_id
    fp_matrix: np.ndarray          # (n_drugs, 2048) float32
    drug_to_idx: dict[str, int]
    cell_order: list[str]
    feature_dims: dict[str, int]   # {"rna": D_rna, "mutations": D_mut}
    name_to_depmap: dict[str, str] # stripped_name.upper() → depmap_id
    key_to_idx: dict[tuple[str, str], int]  # (depmap_id, drug_name) → row in full_df


def load_dataset(processed_dir: Path | None = None) -> DataBundle:
    """Load GDSC2 drug response filtered to cells with RNA + mutations.

    Uses the full drug_response.parquet (not PASO folds) so all three
    split strategies (mixed_set, cell_blind, drug_blind) share the same
    base dataset.
    """
    base = processed_dir or DATA_DIR

    response = pd.read_parquet(base / "drug_response.parquet")
    rna = pd.read_parquet(base / "rna.parquet")
    mutations = pd.read_parquet(base / "mutations.parquet")

    cl_idx = pd.read_parquet(base / "cell_line_index.parquet")
    name_to_depmap: dict[str, str] = {}
    for depmap_id, row in cl_idx.iterrows():
        name_to_depmap[str(row["stripped_name"]).upper()] = str(depmap_id)

    valid_cells = set(rna.index) & set(mutations.index)
    pairs = response[response["depmap_id"].isin(valid_cells)].reset_index(drop=True)
    pairs = pairs.rename(columns={"ln_ic50": "ic50"})

    all_cells = sorted(pairs["depmap_id"].unique())
    all_drugs = sorted(pairs["drug_name"].unique())

    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].to_numpy(dtype=np.float32)
    mut_arr = mutations.loc[all_cells].to_numpy(dtype=np.float32)
    concat_np = np.concatenate([rna_arr, mut_arr], axis=1)

    feature_dims = {"rna": rna_arr.shape[1], "mutations": mut_arr.shape[1]}

    cell_rows = np.array([cell_to_row[c] for c in pairs["depmap_id"]], dtype=np.int32)
    drug_idxs_arr = np.array([drug_to_idx[d] for d in pairs["drug_name"]], dtype=np.int32)
    targets = pairs["ic50"].to_numpy(dtype=np.float32)
    drug_names_arr = pairs["drug_name"].to_numpy()
    cell_ids_arr = pairs["depmap_id"].to_numpy()

    key_to_idx: dict[tuple[str, str], int] = {
        (row["depmap_id"], row["drug_name"]): int(i)
        for i, row in pairs.iterrows()
    }

    from src.data.drug_features import get_drug_fingerprints
    fp_matrix = get_drug_fingerprints(drug_to_idx, base)

    return DataBundle(
        full_df=pairs,
        concat_np=concat_np,
        cell_rows=cell_rows,
        drug_idxs=drug_idxs_arr,
        targets=targets,
        drug_names=drug_names_arr,
        cell_ids=cell_ids_arr,
        fp_matrix=fp_matrix,
        drug_to_idx=drug_to_idx,
        cell_order=all_cells,
        feature_dims=feature_dims,
        name_to_depmap=name_to_depmap,
        key_to_idx=key_to_idx,
    )


def load_dataset_pca(
    rna_dim: int = 550,
    mut_dim: int = 200,
    processed_dir: Path | None = None,
) -> tuple[DataBundle, np.ndarray]:
    """Like load_dataset but returns PCA-compressed omics + full Morgan FP.

    Returns (bundle, pca_concat_np) where pca_concat_np has shape
    (n_cells, rna_dim + mut_dim). The bundle.concat_np remains raw;
    callers should use pca_concat_np for Ridge and MLP comparisons.

    PCA is fit on ALL cells (no train/test leakage for Step 3 where
    PCA is pre-computed once; individual folds z-score normalise after).
    """
    from sklearn.decomposition import PCA

    bundle = load_dataset(processed_dir)
    rna_raw = bundle.concat_np[:, :bundle.feature_dims["rna"]]
    mut_raw = bundle.concat_np[:, bundle.feature_dims["rna"]:]

    n_cells = rna_raw.shape[0]
    n_rna = min(rna_dim, n_cells - 1, rna_raw.shape[1])
    n_mut = min(mut_dim, n_cells - 1, mut_raw.shape[1])

    pca_rna = PCA(n_components=n_rna, random_state=42).fit(rna_raw.astype(np.float64))
    pca_mut = PCA(n_components=n_mut, random_state=42).fit(mut_raw.astype(np.float64))

    rna_c = pca_rna.transform(rna_raw.astype(np.float64)).astype(np.float32)
    mut_c = pca_mut.transform(mut_raw.astype(np.float64)).astype(np.float32)
    pca_concat = np.concatenate([rna_c, mut_c], axis=1)

    return bundle, pca_concat


def make_random_folds(
    full_df: pd.DataFrame,
    split_type: str,
    n_folds: int = 5,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate n_folds using src/data/splits.py with seeds 0..n_folds-1.

    split_type: "mixed_set" | "cell_blind" | "drug_blind"
    Returns list of (train_idx, val_idx, test_idx) over full_df rows.
    """
    from src.data.splits import cell_blind_split, drug_blind_split, mixed_set_split

    fn = {"mixed_set": mixed_set_split,
          "cell_blind": cell_blind_split,
          "drug_blind": drug_blind_split}[split_type]

    pairs = full_df.rename(columns={"drug_name": "drug_name"})  # ensure column name
    folds = []
    for seed in range(n_folds):
        train_idx, val_idx, test_idx = fn(pairs, val_frac=val_frac,
                                          test_frac=test_frac, seed=seed)
        folds.append((train_idx, val_idx, test_idx))
    return folds


def make_paso_drug_blind_folds(
    full_df: pd.DataFrame,
    key_to_idx: dict[tuple[str, str], int],
    name_to_depmap: dict[str, str],
    n_folds: int = 5,
    val_frac: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load PASO pre-generated drug-blind folds and map to full_df indices."""
    from src.utils.paso_folds import load_paso_folds

    raw_folds = load_paso_folds(n_folds)
    result = []
    for train_df, test_df in raw_folds:
        def _map(df: pd.DataFrame) -> np.ndarray:
            idx = []
            for _, row in df.iterrows():
                dep = name_to_depmap.get(str(row["cell_line"]).upper())
                drug = row["drug"]
                if dep and (dep, drug) in key_to_idx:
                    idx.append(key_to_idx[(dep, drug)])
            return np.array(idx, dtype=np.int64)

        full_train = _map(train_df)
        test_idx = _map(test_df)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(full_train))
        n_val = max(1, int(len(full_train) * val_frac))
        val_idx = full_train[perm[:n_val]]
        train_idx = full_train[perm[n_val:]]
        result.append((train_idx, val_idx, test_idx))
    return result


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    drug_names: np.ndarray,
    min_cells: int = 5,
) -> dict:
    """Compute global_r, per_drug_r_mean, gap and n_drugs_evaluated."""
    from scipy.stats import pearsonr
    from src.evaluation.per_drug import per_drug_r

    global_r = float(pearsonr(preds, targets)[0])
    rs = per_drug_r(preds, targets, drug_names, min_cells=min_cells)
    n_drugs = len(rs)
    per_drug_r_mean = float(np.mean(list(rs.values()))) if rs else float("nan")
    return {
        "global_r": round(global_r, 5),
        "per_drug_r_mean": round(per_drug_r_mean, 5),
        "gap": round(per_drug_r_mean - global_r, 5),
        "n_drugs_evaluated": n_drugs,
        "n_drugs_excluded": int((np.unique(drug_names) != None).sum()) - n_drugs,
    }


def bootstrap_ci(
    values: list[float],
    B: int = 1000,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean of values."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    n = len(arr)
    means = np.array([np.mean(rng.choice(arr, n, replace=True)) for _ in range(B)])
    lo = (1 - level) / 2 * 100
    hi = (1 + level) / 2 * 100
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def paired_ttest(values: list[float]) -> tuple[float, float]:
    """One-sample t-test against H0: mean = 0. Returns (t_stat, p_value)."""
    from scipy.stats import ttest_1samp
    t, p = ttest_1samp(values, 0)
    return float(t), float(p)
