"""07_cross_dataset_transfer: GDSC2 → PRISM cross-dataset transfer Ridge ablation.

Train Ridge on all GDSC2 pairs (IC50), evaluate on PRISM Repurposing pairs (AUC).
Conditions: morgan_fp vs no_drug. Single train/test split (no CV).

This is the strongest drug-feature null test: structural features must generalize
across dataset boundaries to show signal here.

Output: report/data/metrics.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

from sklearn.decomposition import PCA

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

RIDGE_ALPHA = 1.0
MIN_CELLS_PER_DRUG = 50
MIN_CELLS_EVAL = 5
RNA_DIM = 550
MUT_DIM = 200


def load_gdsc2(data_dir: Path) -> pd.DataFrame:
    """Load GDSC2 drug response. Returns DataFrame with depmap_id, drug_name, response."""
    dr = pd.read_parquet(data_dir / "drug_response.parquet")
    logger.info("GDSC2 raw: %d pairs, %d drugs, %d cells",
                len(dr), dr["drug_name"].nunique(), dr["depmap_id"].nunique())
    return dr


def build_fp_matrix(drug_names: List[str], data_dir: Path) -> np.ndarray:
    """Build Morgan FP matrix (n_drugs, 2048) for a given drug list."""
    drug_to_idx: Dict[str, int] = {d: i for i, d in enumerate(drug_names)}
    return get_drug_fingerprints(drug_to_idx, data_dir)


def main() -> None:
    t_start = time.perf_counter()
    logs_dir = EXP_DIR / "logs"
    report_dir = EXP_DIR / "report" / "data"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(logs_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("07_cross_dataset_transfer: GDSC2 → PRISM Ridge ablation (morgan_fp vs no_drug)")

    # -------------------------------------------------------------------------
    # Load cell features
    # -------------------------------------------------------------------------
    logger.info("Loading omics data...")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    available_cells = set(rna.index) & set(mutations.index)
    logger.info("RNA: %s  Mutations: %s  available cells: %d",
                rna.shape, mutations.shape, len(available_cells))

    # -------------------------------------------------------------------------
    # Load GDSC2 (train)
    # -------------------------------------------------------------------------
    gdsc_raw = load_gdsc2(DATA_DIR)
    gdsc_df = gdsc_raw[gdsc_raw["depmap_id"].isin(available_cells)].copy()
    gdsc_drugs = sorted(gdsc_df["drug_name"].unique())
    gdsc_cells = sorted(gdsc_df["depmap_id"].unique())
    logger.info("GDSC2 train: %d pairs, %d drugs, %d cells",
                len(gdsc_df), len(gdsc_drugs), len(gdsc_cells))

    # -------------------------------------------------------------------------
    # Load PRISM (test)
    # -------------------------------------------------------------------------
    prism_raw = load_prism(DATA_DIR)
    prism_df, n_prism_drugs, n_prism_cells = preprocess_prism(prism_raw, available_cells)
    prism_cells = sorted(prism_df["depmap_id"].unique())
    logger.info("PRISM test: %d pairs, %d drugs, %d cells",
                len(prism_df), n_prism_drugs, n_prism_cells)

    # Filter PRISM drugs to those with enough cell lines
    drug_cell_counts = prism_df.groupby("drug_name")["depmap_id"].nunique()
    valid_prism_drugs = sorted(drug_cell_counts[drug_cell_counts >= MIN_CELLS_PER_DRUG].index)
    prism_df = prism_df[prism_df["drug_name"].isin(valid_prism_drugs)].copy()
    logger.info("PRISM after MIN_CELLS_PER_DRUG=%d filter: %d drugs",
                MIN_CELLS_PER_DRUG, len(valid_prism_drugs))

    # Drug overlap diagnostics
    gdsc_drug_set = set(gdsc_drugs)
    prism_drug_set = set(valid_prism_drugs)
    overlap = gdsc_drug_set & prism_drug_set
    logger.info(
        "Drug overlap: GDSC2=%d, PRISM=%d, overlap=%d (%.1f%% of PRISM)",
        len(gdsc_drug_set), len(prism_drug_set), len(overlap),
        100 * len(overlap) / max(1, len(prism_drug_set)),
    )

    # Validation checks
    assert len(gdsc_cells) >= 400, f"Too few GDSC2 train cells: {len(gdsc_cells)}"
    assert len(prism_cells) >= 200, f"Too few PRISM test cells: {len(prism_cells)}"
    assert len(valid_prism_drugs) >= 100, f"Too few PRISM test drugs: {len(valid_prism_drugs)}"

    # -------------------------------------------------------------------------
    # Build cell feature matrices
    # -------------------------------------------------------------------------
    all_cells = sorted(set(gdsc_cells) | set(prism_cells))
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)

    # PCA fit on GDSC2 training cells only
    train_cell_rows = np.array([cell_to_row[c] for c in gdsc_cells], dtype=np.int32)
    rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_cell_rows,
                                     rna_dim=RNA_DIM, mut_dim=MUT_DIM)
    cell_feat = np.concatenate([rna_pca, mut_pca], axis=1)
    logger.info("Cell features (PCA): %s", cell_feat.shape)

    # --- PCA variance preservation check for PRISM cells ---
    # PCA was fit on GDSC2 cells; verify it captures adequate variance in PRISM cells.
    # Low preservation (< 60%) means PRISM cell features are degraded and results are unreliable.
    prism_cell_rows_all = np.array([cell_to_row[c] for c in prism_cells], dtype=np.int32)
    gdsc_only_rows = train_cell_rows
    n_rna_comp = min(RNA_DIM, len(np.unique(gdsc_only_rows)) - 1, rna_arr.shape[1])
    n_mut_comp = min(MUT_DIM, len(np.unique(gdsc_only_rows)) - 1, mut_arr.shape[1])
    pca_rna_obj = PCA(n_components=n_rna_comp, random_state=42)
    pca_rna_obj.fit(rna_arr[np.unique(gdsc_only_rows)].astype(np.float64))
    pca_mut_obj = PCA(n_components=n_mut_comp, random_state=42)
    pca_mut_obj.fit(mut_arr[np.unique(gdsc_only_rows)].astype(np.float64))

    prism_rna_raw = rna_arr[prism_cell_rows_all].astype(np.float64)
    prism_mut_raw = mut_arr[prism_cell_rows_all].astype(np.float64)
    rna_var_ratio = float(pca_rna_obj.explained_variance_ratio_.sum())
    mut_var_ratio = float(pca_mut_obj.explained_variance_ratio_.sum())

    # Fraction of PRISM total variance explained by GDSC2-fit PCA
    prism_rna_proj = pca_rna_obj.transform(prism_rna_raw)
    prism_rna_recon = pca_rna_obj.inverse_transform(prism_rna_proj)
    rna_prism_var_preserved = float(
        1.0 - np.var(prism_rna_raw - prism_rna_recon) / (np.var(prism_rna_raw) + 1e-12)
    )
    prism_mut_proj = pca_mut_obj.transform(prism_mut_raw)
    prism_mut_recon = pca_mut_obj.inverse_transform(prism_mut_proj)
    mut_prism_var_preserved = float(
        1.0 - np.var(prism_mut_raw - prism_mut_recon) / (np.var(prism_mut_raw) + 1e-12)
    )
    logger.info(
        "PCA variance preserved in GDSC2 train cells: RNA=%.1f%%  mut=%.1f%%",
        100 * rna_var_ratio, 100 * mut_var_ratio,
    )
    logger.info(
        "PCA variance preserved in PRISM cells: RNA=%.1f%%  mut=%.1f%%",
        100 * rna_prism_var_preserved, 100 * mut_prism_var_preserved,
    )
    if rna_prism_var_preserved < 0.60 or mut_prism_var_preserved < 0.60:
        logger.warning(
            "LOW PRISM VARIANCE PRESERVATION (< 60%%): "
            "RNA=%.1f%%  mut=%.1f%% — cross-dataset results may be unreliable.",
            100 * rna_prism_var_preserved, 100 * mut_prism_var_preserved,
        )

    # Training cell/drug arrays
    gdsc_cell_idx = np.array([cell_to_row[c] for c in gdsc_df["depmap_id"]], dtype=np.int32)
    y_train = gdsc_df["ln_ic50"].values.astype(np.float32)
    X_cell_train = cell_feat[gdsc_cell_idx]

    # Test cell arrays
    prism_cell_idx = np.array([cell_to_row[c] for c in prism_df["depmap_id"]], dtype=np.int32)
    y_test = prism_df["response"].values.astype(np.float32)
    test_drug_names = prism_df["drug_name"].values
    X_cell_test = cell_feat[prism_cell_idx]

    # -------------------------------------------------------------------------
    # Drug fingerprints (separate index for GDSC2 and PRISM)
    # -------------------------------------------------------------------------
    gdsc_drug_to_idx = {d: i for i, d in enumerate(gdsc_drugs)}
    prism_drug_to_idx = {d: i for i, d in enumerate(valid_prism_drugs)}

    gdsc_fp = get_drug_fingerprints(gdsc_drug_to_idx, DATA_DIR)
    prism_fp = get_drug_fingerprints(prism_drug_to_idx, DATA_DIR)
    logger.info("GDSC2 FP: %s (nonzero=%d)  PRISM FP: %s (nonzero=%d)",
                gdsc_fp.shape, int((gdsc_fp.sum(axis=1) > 0).sum()),
                prism_fp.shape, int((prism_fp.sum(axis=1) > 0).sum()))

    gdsc_drug_idx_arr = np.array([gdsc_drug_to_idx[d] for d in gdsc_df["drug_name"]], dtype=np.int32)
    prism_drug_idx_arr = np.array([prism_drug_to_idx[d] for d in prism_df["drug_name"]], dtype=np.int32)

    # -------------------------------------------------------------------------
    # Condition: no_drug (cell features only)
    # -------------------------------------------------------------------------
    logger.info("=== Condition: no_drug ===")
    t0 = time.perf_counter()
    sc_nd = safe_fit_scaler(X_cell_train)
    X_tr_nd = sc_nd.transform(X_cell_train)
    X_te_nd = sc_nd.transform(X_cell_test)
    ridge_nd = Ridge(alpha=RIDGE_ALPHA)
    ridge_nd.fit(X_tr_nd, y_train)
    preds_nd = ridge_nd.predict(X_te_nd)
    r_no_drug = mean_per_drug_r(preds_nd, y_test, test_drug_names, min_cells=MIN_CELLS_EVAL)
    logger.info("no_drug: per-drug r = %.4f  (%.1fs)", r_no_drug, time.perf_counter() - t0)

    # -------------------------------------------------------------------------
    # Condition: morgan_fp (cell + drug fingerprint)
    # -------------------------------------------------------------------------
    logger.info("=== Condition: morgan_fp ===")
    t0 = time.perf_counter()
    X_drug_train = gdsc_fp[gdsc_drug_idx_arr]
    X_drug_test = prism_fp[prism_drug_idx_arr]
    X_tr_fp = np.concatenate([X_cell_train, X_drug_train], axis=1)
    X_te_fp = np.concatenate([X_cell_test, X_drug_test], axis=1)
    sc_fp = safe_fit_scaler(X_tr_fp)
    X_tr_fp = sc_fp.transform(X_tr_fp)
    X_te_fp = sc_fp.transform(X_te_fp)
    ridge_fp = Ridge(alpha=RIDGE_ALPHA)
    ridge_fp.fit(X_tr_fp, y_train)
    preds_fp = ridge_fp.predict(X_te_fp)
    r_morgan_fp = mean_per_drug_r(preds_fp, y_test, test_drug_names, min_cells=MIN_CELLS_EVAL)
    logger.info("morgan_fp: per-drug r = %.4f  (%.1fs)", r_morgan_fp, time.perf_counter() - t0)

    delta = r_morgan_fp - r_no_drug
    elapsed = time.perf_counter() - t_start

    logger.info("=" * 60)
    logger.info("GDSC2 → PRISM cross-dataset transfer:")
    logger.info("  no_drug:   per-drug r = %.4f", r_no_drug)
    logger.info("  morgan_fp: per-drug r = %.4f", r_morgan_fp)
    logger.info("  delta: %.4f  (positive = morgan_fp > no_drug)", delta)
    logger.info("  n_train_pairs=%d  n_test_pairs=%d  n_test_drugs=%d  n_test_cells=%d",
                len(gdsc_df), len(prism_df), len(valid_prism_drugs), len(prism_cells))
    logger.info("  drug_overlap=%d (%.1f%% of PRISM)",
                len(overlap), 100 * len(overlap) / max(1, len(prism_drug_set)))
    logger.info("  elapsed=%.1fs", elapsed)

    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    output = {
        "no_drug": float(r_no_drug),
        "morgan_fp": float(r_morgan_fp),
        "delta_morgan_vs_no_drug": float(delta),
        "n_train_pairs": len(gdsc_df),
        "n_test_pairs": len(prism_df),
        "n_train_drugs": len(gdsc_drugs),
        "n_test_drugs": len(valid_prism_drugs),
        "n_train_cells": len(gdsc_cells),
        "n_test_cells": len(prism_cells),
        "drug_overlap_count": len(overlap),
        "drug_overlap_pct_prism": float(100 * len(overlap) / max(1, len(prism_drug_set))),
        "train_dataset": "gdsc2",
        "test_dataset": "prism_repurposing",
        "pca_variance_preserved": {
            "gdsc2_train_rna": round(rna_var_ratio, 4),
            "gdsc2_train_mut": round(mut_var_ratio, 4),
            "prism_test_rna": round(rna_prism_var_preserved, 4),
            "prism_test_mut": round(mut_prism_var_preserved, 4),
        },
    }
    out_path = report_dir / "metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
