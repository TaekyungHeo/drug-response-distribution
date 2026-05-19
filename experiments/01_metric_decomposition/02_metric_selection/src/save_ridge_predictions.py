"""Step 1: Save Ridge drug_blind fold predictions to parquet.

Runs Ridge with nested-CV alpha selection (grid [0.01, 0.1, 1, 10, 100, 1000],
matching 01_global_vs_perdrug/src/runner.py) on PASO drug_blind 5-fold splits.
Saves raw (depmap_id, drug_name, y_true, y_pred) per fold.

Runtime: ~10 min CPU.
Output:  results/fold_predictions/ridge_drug_blind_fold{0..4}.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parents[4]
EXP_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = EXP_DIR / "results" / "fold_predictions"

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RNA_DIM = 550
MUT_DIM = 200
N_FOLDS = 5
VAL_FRAC = 0.1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Load GDSC2 drug response with PCA-compressed omics + Morgan FP features.

    Returns:
        pairs: DataFrame [depmap_id, drug_name, ic50]
        X_all: (n_pairs, rna_dim+mut_dim+2048) float32 feature matrix
        y_all: (n_pairs,) float32 ln_IC50 targets
        drug_names: (n_pairs,) str
        name_to_depmap: stripped_name.upper() → depmap_id (for PASO fold matching)
    """
    from scipy.stats import pearsonr as _  # noqa: F401 — verify scipy available
    from src.data.drug_features import get_drug_fingerprints

    log.info("Loading drug response + omics data...")
    response = pd.read_parquet(DATA_DIR / "drug_response.parquet")
    rna_df = pd.read_parquet(DATA_DIR / "rna.parquet")
    mut_df = pd.read_parquet(DATA_DIR / "mutations.parquet")
    cell_index = pd.read_parquet(DATA_DIR / "cell_line_index.parquet")

    name_to_depmap: dict[str, str] = {
        str(row["stripped_name"]).upper(): str(depmap_id)
        for depmap_id, row in cell_index.iterrows()
    }

    valid_cells = set(rna_df.index) & set(mut_df.index)
    pairs = response[response["depmap_id"].isin(valid_cells)].reset_index(drop=True)
    pairs = pairs.rename(columns={"ln_ic50": "ic50"})

    all_cells = sorted(pairs["depmap_id"].unique())
    all_drugs = sorted(pairs["drug_name"].unique())
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    log.info("Cells: %d | Drugs: %d | Pairs: %d", len(all_cells), len(all_drugs), len(pairs))

    # PCA compression fit on all cells (consistent with 01_global_vs_perdrug)
    rna_arr = rna_df.loc[all_cells].to_numpy(dtype=np.float64)
    mut_arr = mut_df.loc[all_cells].to_numpy(dtype=np.float64)
    n_rna = min(RNA_DIM, len(all_cells) - 1, rna_arr.shape[1])
    n_mut = min(MUT_DIM, len(all_cells) - 1, mut_arr.shape[1])
    rna_c = PCA(n_components=n_rna, random_state=42).fit_transform(rna_arr).astype(np.float32)
    mut_c = PCA(n_components=n_mut, random_state=42).fit_transform(mut_arr).astype(np.float32)
    omics = np.concatenate([rna_c, mut_c], axis=1)  # (n_cells, rna_dim+mut_dim)

    fp_matrix = get_drug_fingerprints(drug_to_idx)  # (n_drugs, 2048)

    cell_rows = np.array([cell_to_row[c] for c in pairs["depmap_id"]], dtype=np.int32)
    drug_idxs = np.array([drug_to_idx[d] for d in pairs["drug_name"]], dtype=np.int32)

    X_all = np.concatenate([omics[cell_rows], fp_matrix[drug_idxs]], axis=1).astype(np.float32)
    y_all = pairs["ic50"].to_numpy(dtype=np.float32)
    drug_names = pairs["drug_name"].to_numpy()

    log.info("Feature matrix: %s (%.1f MB)", X_all.shape, X_all.nbytes / 1e6)
    return pairs, X_all, y_all, drug_names, name_to_depmap


def make_paso_folds(
    pairs: pd.DataFrame,
    name_to_depmap: dict[str, str],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load PASO drug_blind folds and map to row indices in pairs."""
    from src.utils.paso_folds import load_paso_folds

    key_to_idx: dict[tuple[str, str], int] = {
        (row["depmap_id"], row["drug_name"]): i
        for i, row in pairs.iterrows()
    }

    raw_folds = load_paso_folds(N_FOLDS)
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

        all_train = _map(train_df)
        test_idx = _map(test_df)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(all_train))
        n_val = max(1, int(len(all_train) * VAL_FRAC))
        result.append((all_train[perm[n_val:]], all_train[perm[:n_val]], test_idx))

    log.info("Folds: %d | test sizes: %s", len(result), [len(f[2]) for f in result])
    return result


def run_ridge_fold(
    X_all: np.ndarray,
    y_all: np.ndarray,
    drug_names: np.ndarray,
    depmap_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_i: int,
) -> pd.DataFrame:
    from scipy.stats import pearsonr

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all[train_idx])
    X_va = sc.transform(X_all[val_idx])
    X_te = sc.transform(X_all[test_idx])

    best_alpha, best_val_r = ALPHA_GRID[0], -np.inf
    for alpha in ALPHA_GRID:
        preds_va = Ridge(alpha=alpha).fit(X_tr, y_all[train_idx]).predict(X_va)
        val_r = float(pearsonr(preds_va, y_all[val_idx])[0])
        if val_r > best_val_r:
            best_val_r, best_alpha = val_r, alpha

    preds_te = Ridge(alpha=best_alpha).fit(X_tr, y_all[train_idx]).predict(X_te)
    log.info("Fold %d | alpha=%.3g | val_r=%.4f | test_n=%d", fold_i, best_alpha, best_val_r, len(test_idx))

    return pd.DataFrame({
        "depmap_id": depmap_ids[test_idx],
        "drug_name": drug_names[test_idx],
        "y_true": y_all[test_idx].astype(np.float32),
        "y_pred": preds_te.astype(np.float32),
    })


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Fold index 0-4. Defaults to SLURM_ARRAY_TASK_ID if set.",
    )
    args = parser.parse_args()

    fold_i = args.fold
    if fold_i is None:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is not None:
            fold_i = int(task_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs, X_all, y_all, drug_names, name_to_depmap = load_data()
    depmap_ids = pairs["depmap_id"].to_numpy()
    folds = make_paso_folds(pairs, name_to_depmap)

    fold_range = [fold_i] if fold_i is not None else range(N_FOLDS)

    for i in fold_range:
        out_path = RESULTS_DIR / f"ridge_drug_blind_fold{i}.parquet"
        if out_path.exists():
            log.info("Fold %d already exists, skipping.", i)
            continue
        train_idx, val_idx, test_idx = folds[i]
        df = run_ridge_fold(X_all, y_all, drug_names, depmap_ids,
                            train_idx, val_idx, test_idx, i)
        df.to_parquet(out_path, index=False)
        log.info("Saved → %s (%d rows)", out_path.name, len(df))

    log.info("Done.")


if __name__ == "__main__":
    main()
