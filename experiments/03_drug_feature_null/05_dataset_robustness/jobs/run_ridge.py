"""05_dataset_robustness/run_ridge.py: PRISM dataset Ridge ablation (morgan_fp vs no_drug).

5-fold drug-blind CV on the PRISM Repurposing dataset. Tests whether the drug-feature
null holds at 3× the drug count (1,415 drugs vs 233 in GDSC2) and a different assay
platform.

Data: data/processed/prism_drug_response.parquet
  - columns: depmap_id (str), drug_name (str), ln_ic50 (float32)
  - This is the canonical preprocessed PRISM file used across the project.
  - The spec referenced 'prism_repurposing.parquet' with 'auc' and 'broad_id' columns,
    but the actual file has ln_ic50 and drug_name. This script uses the real file.
  - Fallback: if prism_drug_response.parquet is absent, attempt to read from
    data/external/prism/repurposing_secondary_screen_dose_response.csv (DepMap portal).

Output: report/data/metrics.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

from src.data.drug_features import get_drug_fingerprints
from src.data.prism import load_prism, preprocess_prism
from src.evaluation.per_drug import mean_per_drug_r
from src.utils.ridge import compress_cell, safe_fit_scaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "processed"
EXP_DIR = Path(__file__).parents[1]
K_FOLDS = 5
MIN_CELLS_PER_DRUG = 50
MIN_CELLS_EVAL = 5
RIDGE_ALPHA = 1.0
FOLD_SEED = 42


# ---------------------------------------------------------------------------
# Drug-blind fold assignment
# ---------------------------------------------------------------------------

def make_drug_folds(drug_names: List[str], n_folds: int = 5, seed: int = FOLD_SEED) -> List[List[str]]:
    """Randomly assign drugs to n_folds folds."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(drug_names))
    fold_assignments = np.array_split(shuffled, n_folds)
    return [[drug_names[i] for i in fold_idx] for fold_idx in fold_assignments]


# ---------------------------------------------------------------------------
# Ridge fold runner
# ---------------------------------------------------------------------------

def run_fold(
    k: int,
    fold_drugs: List[List[str]],
    dr: pd.DataFrame,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    fp_matrix: np.ndarray,
    drug_to_idx: Dict[str, int],
) -> Tuple[float, float]:
    """Fit Ridge for fold k, return (no_drug_r, morgan_fp_r)."""
    test_drug_set = set(fold_drugs[k])
    train_drug_set = set(d for i, drugs in enumerate(fold_drugs) if i != k for d in drugs)

    available_cells = set(rna.index) & set(mutations.index)

    train_df = dr[dr["drug_name"].isin(train_drug_set) & dr["depmap_id"].isin(available_cells)].copy()
    test_df = dr[dr["drug_name"].isin(test_drug_set) & dr["depmap_id"].isin(available_cells)].copy()

    # Exclude test drugs with < MIN_CELLS_PER_DRUG test pairs
    test_counts = test_df.groupby("drug_name").size()
    valid_test_drugs = test_counts[test_counts >= MIN_CELLS_PER_DRUG].index
    test_df = test_df[test_df["drug_name"].isin(valid_test_drugs)].copy()

    logger.info(
        "  Fold %d: train_drugs=%d test_drugs=%d train_pairs=%d test_pairs=%d",
        k, len(train_drug_set), len(valid_test_drugs), len(train_df), len(test_df),
    )

    all_cells_train = sorted(train_df["depmap_id"].unique())
    all_cells_test = sorted(test_df["depmap_id"].unique())
    all_cells = sorted(set(all_cells_train) | set(all_cells_test))

    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)

    # PCA fit on training cells only
    train_cell_rows = np.array([cell_to_row[c] for c in all_cells_train], dtype=np.int32)
    rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_cell_rows, rna_dim=550, mut_dim=200)
    cell_feat = np.concatenate([rna_pca, mut_pca], axis=1)

    train_cell_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
    test_cell_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
    train_drug_idx = np.array([drug_to_idx[d] for d in train_df["drug_name"]], dtype=np.int32)
    test_drug_idx = np.array([drug_to_idx[d] for d in test_df["drug_name"]], dtype=np.int32)
    y_train = train_df["response"].values.astype(np.float32)
    y_test = test_df["response"].values.astype(np.float32)
    test_drug_names = test_df["drug_name"].values

    X_cell_train = cell_feat[train_cell_idx]
    X_cell_test = cell_feat[test_cell_idx]

    # --- no_drug ---
    sc_nd = safe_fit_scaler(X_cell_train)
    X_tr_nd = sc_nd.transform(X_cell_train)
    X_te_nd = sc_nd.transform(X_cell_test)
    ridge_nd = Ridge(alpha=RIDGE_ALPHA)
    ridge_nd.fit(X_tr_nd, y_train)
    preds_nd = ridge_nd.predict(X_te_nd)
    r_no_drug = mean_per_drug_r(preds_nd, y_test, test_drug_names, min_cells=MIN_CELLS_EVAL)

    # --- morgan_fp ---
    X_drug_train = fp_matrix[train_drug_idx]
    X_drug_test = fp_matrix[test_drug_idx]
    X_tr_fp = np.concatenate([X_cell_train, X_drug_train], axis=1)
    X_te_fp = np.concatenate([X_cell_test, X_drug_test], axis=1)
    sc_fp = safe_fit_scaler(X_tr_fp)
    X_tr_fp = sc_fp.transform(X_tr_fp)
    X_te_fp = sc_fp.transform(X_te_fp)
    ridge_fp = Ridge(alpha=RIDGE_ALPHA)
    ridge_fp.fit(X_tr_fp, y_train)
    preds_fp = ridge_fp.predict(X_te_fp)
    r_morgan_fp = mean_per_drug_r(preds_fp, y_test, test_drug_names, min_cells=MIN_CELLS_EVAL)

    logger.info("  Fold %d: no_drug=%.4f  morgan_fp=%.4f  delta=%.4f",
                k, r_no_drug, r_morgan_fp, r_morgan_fp - r_no_drug)
    return r_no_drug, r_morgan_fp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("05_dataset_robustness/run_ridge: PRISM 5-fold drug-blind Ridge ablation")

    # Load PRISM data
    df_raw = load_prism(DATA_DIR)

    # Load RNA (for cell intersection)
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  Mutations: %s", rna.shape, mutations.shape)

    # Filter to cells that have both RNA and mutations
    rna_cell_ids = set(rna.index) & set(mutations.index)

    dr, n_drugs, n_cells = preprocess_prism(df_raw, rna_cell_ids)

    # Build drug index
    drug_names_all = sorted(dr["drug_name"].unique())
    drug_to_idx = {d: i for i, d in enumerate(drug_names_all)}

    # Morgan fingerprints
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
    logger.info("Fingerprint matrix: %s  nonzero rows=%d",
                fp_matrix.shape, int((fp_matrix.sum(axis=1) > 0).sum()))

    # Random 5-fold drug assignment
    fold_drugs = make_drug_folds(drug_names_all, n_folds=K_FOLDS, seed=FOLD_SEED)
    logger.info("Fold sizes: %s", [len(f) for f in fold_drugs])

    # 5-fold CV
    no_drug_folds: List[float] = []
    morgan_fp_folds: List[float] = []

    for k in range(K_FOLDS):
        logger.info("=== Fold %d/%d ===", k, K_FOLDS - 1)
        r_nd, r_fp = run_fold(k, fold_drugs, dr, rna, mutations, fp_matrix, drug_to_idx)
        no_drug_folds.append(r_nd)
        morgan_fp_folds.append(r_fp)

    # Summary
    nd_mean = float(np.mean(no_drug_folds))
    nd_std = float(np.std(no_drug_folds))
    fp_mean = float(np.mean(morgan_fp_folds))
    fp_std = float(np.std(morgan_fp_folds))
    delta = fp_mean - nd_mean

    logger.info("=" * 60)
    logger.info("no_drug:   mean=%.4f ± %.4f  folds=%s", nd_mean, nd_std,
                [round(r, 4) for r in no_drug_folds])
    logger.info("morgan_fp: mean=%.4f ± %.4f  folds=%s  delta=%.4f", fp_mean, fp_std,
                [round(r, 4) for r in morgan_fp_folds], delta)

    # Write output
    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "no_drug": {
            "mean": nd_mean,
            "std": nd_std,
            "folds": [float(r) for r in no_drug_folds],
        },
        "morgan_fp": {
            "mean": fp_mean,
            "std": fp_std,
            "folds": [float(r) for r in morgan_fp_folds],
            "delta": delta,
        },
        "n_drugs": n_drugs,
        "n_cells": n_cells,
        "dataset": "prism_repurposing",
    }
    out_path = report_dir / "metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
