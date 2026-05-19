"""MoA one-hot as feature control experiment.

Tests whether appending MoA class identity as a one-hot feature to Ridge input
improves per-drug r. Two conditions:
  - baseline: cell features only (RNA PCA 550 + mutation PCA 200)
  - onehot:   cell features + MoA one-hot encoding per drug

PASO 10-fold drug-blind CV, Ridge(alpha=1.0).
Reports per-drug r (primary) and global r (diagnostic) for both conditions.

CLI:
  --smoke    run only 2 folds (fast check)

Output: report/data/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.evaluation.per_drug import per_drug_r
from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell
from src.utils.solutions import load_moa_annotations

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
RIDGE_ALPHA = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_moa_onehot(
    drug_to_idx: dict[str, int],
    moa: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    """Build a (n_drugs, n_moa_classes) one-hot matrix.

    Drugs absent from *moa* get an all-zero row.

    Returns:
        (onehot, moa_classes) where onehot is float32 and moa_classes are
        the sorted unique pathway labels.
    """
    moa_classes = sorted(set(moa.values()))
    class_to_col = {c: i for i, c in enumerate(moa_classes)}
    n_drugs = len(drug_to_idx)
    n_classes = len(moa_classes)
    oh = np.zeros((n_drugs, n_classes), dtype=np.float32)
    for drug, idx in drug_to_idx.items():
        pathway = moa.get(drug)
        if pathway is not None and pathway in class_to_col:
            oh[idx, class_to_col[pathway]] = 1.0
    return oh, moa_classes


def main() -> None:
    parser = argparse.ArgumentParser(description="MoA one-hot control experiment")
    parser.add_argument("--smoke", action="store_true", help="Run only 2 folds")
    args = parser.parse_args()

    n_folds = 2 if args.smoke else K_FOLDS
    logger.info("03_onehot_control | ROOT=%s | folds=%d", ROOT, n_folds)

    # ---- Load omics ----
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    available_cells = set(rna.index) & set(mutations.index)

    # ---- Build drug index from all PASO folds ----
    all_drugs: set[str] = set()
    for k in range(K_FOLDS):
        tr = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{k}.csv")
        te = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{k}.csv")
        all_drugs |= set(tr["drug"].unique()) | set(te["drug"].unique())
    drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(sorted(all_drugs))}
    logger.info("PASO drug set: %d drugs", len(drug_to_idx))

    # ---- MoA annotations & one-hot ----
    moa = load_moa_annotations()
    logger.info("MoA annotations loaded: %d drugs annotated", len(moa))

    onehot, moa_classes = build_moa_onehot(drug_to_idx, moa)
    n_annotated = int((onehot.sum(axis=1) > 0).sum())
    logger.info(
        "MoA one-hot: %d classes, %d/%d drugs annotated",
        len(moa_classes), n_annotated, len(drug_to_idx),
    )

    # ---- Run folds: both conditions per fold ----
    pooled_baseline: dict[str, float] = {}
    pooled_onehot: dict[str, float] = {}
    # For global r: accumulate predictions and targets across folds
    global_preds_base: list[np.ndarray] = []
    global_preds_oh: list[np.ndarray] = []
    global_targets: list[np.ndarray] = []

    for fold_i in range(n_folds):
        t0 = datetime.now()
        logger.info("Fold %d/%d started at %s", fold_i, n_folds, t0.strftime("%H:%M:%S"))

        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
        )
        train_df = pd.DataFrame(train_df[train_df["drug_name"].isin(drug_to_idx)])
        test_df = pd.DataFrame(test_df[test_df["drug_name"].isin(drug_to_idx)])

        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty train or test — skipping", fold_i)
            continue

        # Cell features (shared across both conditions)
        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        cell_to_row = {c: i for i, c in enumerate(all_cells)}
        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        train_cell_rows = np.array(
            [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
        )
        train_cell_set = np.unique(train_cell_rows)
        rna_c, mut_c = compress_cell(rna_arr, mut_arr, train_cell_set)
        cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

        # Build pair matrices
        train_rows_c = np.array(
            [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
        )
        test_rows_c = np.array(
            [cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32
        )
        X_train_base = cell_feat[train_rows_c]
        X_test_base = cell_feat[test_rows_c]
        y_train = train_df["ln_ic50"].values.astype(np.float32)
        y_test = test_df["ln_ic50"].values.astype(np.float32)
        drug_names_test = test_df["drug_name"].values

        # MoA one-hot per sample
        train_drug_idxs = np.array(
            [drug_to_idx[d] for d in train_df["drug_name"]], dtype=np.int32
        )
        test_drug_idxs = np.array(
            [drug_to_idx[d] for d in test_df["drug_name"]], dtype=np.int32
        )
        X_train_oh = np.hstack([X_train_base, onehot[train_drug_idxs]])
        X_test_oh = np.hstack([X_test_base, onehot[test_drug_idxs]])

        # Validation: check one-hot columns are non-zero in training data
        oh_train_block = onehot[train_drug_idxs]
        oh_col_sums = oh_train_block.sum(axis=0)
        n_active_cols = int((oh_col_sums > 0).sum())
        logger.info(
            "  Fold %d: onehot train block — %d/%d MoA columns active, "
            "min_sum=%.0f max_sum=%.0f",
            fold_i, n_active_cols, len(moa_classes),
            oh_col_sums.min(), oh_col_sums.max(),
        )

        logger.info(
            "  Fold %d: train=%d test=%d base_feat=%d oh_feat=%d",
            fold_i, len(y_train), len(y_test),
            X_train_base.shape[1], X_train_oh.shape[1],
        )

        # ---- Condition 1: baseline (cell features only) ----
        model_base = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model_base.fit(X_train_base.astype(np.float64), y_train.astype(np.float64))
        preds_base = model_base.predict(X_test_base.astype(np.float64)).astype(np.float32)

        fold_base = per_drug_r(preds_base, y_test, drug_names_test, min_cells=5)
        pooled_baseline.update(fold_base)

        # ---- Condition 2: onehot (cell features + MoA one-hot) ----
        model_oh = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model_oh.fit(X_train_oh.astype(np.float64), y_train.astype(np.float64))
        preds_oh = model_oh.predict(X_test_oh.astype(np.float64)).astype(np.float32)

        fold_oh = per_drug_r(preds_oh, y_test, drug_names_test, min_cells=5)
        pooled_onehot.update(fold_oh)

        # Accumulate for global r
        global_preds_base.append(preds_base)
        global_preds_oh.append(preds_oh)
        global_targets.append(y_test)

        t1 = datetime.now()
        elapsed = (t1 - t0).total_seconds()
        base_mean = float(np.mean(list(fold_base.values()))) if fold_base else float("nan")
        oh_mean = float(np.mean(list(fold_oh.values()))) if fold_oh else float("nan")
        logger.info(
            "  Fold %d: base_r=%.4f oh_r=%.4f n_drugs=%d elapsed=%.1fs",
            fold_i, base_mean, oh_mean, len(fold_base), elapsed,
        )

    # ---- Aggregate per-drug r ----
    common_drugs = sorted(set(pooled_baseline.keys()) & set(pooled_onehot.keys()))
    base_rs = np.array([pooled_baseline[d] for d in common_drugs])
    oh_rs = np.array([pooled_onehot[d] for d in common_drugs])
    mean_base_r = float(np.mean(base_rs)) if len(base_rs) > 0 else float("nan")
    mean_oh_r = float(np.mean(oh_rs)) if len(oh_rs) > 0 else float("nan")

    # ---- Global r ----
    all_preds_base = np.concatenate(global_preds_base)
    all_preds_oh = np.concatenate(global_preds_oh)
    all_targets = np.concatenate(global_targets)
    global_r_base = float(pearsonr(all_preds_base, all_targets)[0])
    global_r_oh = float(pearsonr(all_preds_oh, all_targets)[0])

    delta_per_drug = mean_oh_r - mean_base_r
    delta_global = global_r_oh - global_r_base

    logger.info("=" * 70)
    logger.info("RESULTS (%d drugs, %d folds)", len(common_drugs), n_folds)
    logger.info(
        "  Baseline:  per_drug_r=%.4f  global_r=%.4f",
        mean_base_r, global_r_base,
    )
    logger.info(
        "  Onehot:    per_drug_r=%.4f  global_r=%.4f",
        mean_oh_r, global_r_oh,
    )
    logger.info(
        "  Delta:     per_drug_r=%+.4f  global_r=%+.4f",
        delta_per_drug, delta_global,
    )

    # Conclusion
    if abs(delta_per_drug) <= 0.002:
        conclusion = "MoA as representation does not improve per-drug r"
    elif delta_per_drug > 0.002:
        conclusion = "MoA as representation IMPROVES per-drug r (unexpected)"
    else:
        conclusion = "MoA as representation HURTS per-drug r"
    logger.info("  Conclusion: %s", conclusion)

    # ---- Per-drug detail ----
    per_drug_list: list[dict] = []
    for drug in common_drugs:
        per_drug_list.append({
            "drug": drug,
            "baseline_r": round(pooled_baseline[drug], 6),
            "onehot_r": round(pooled_onehot[drug], 6),
            "delta": round(pooled_onehot[drug] - pooled_baseline[drug], 6),
        })

    # ---- Write results ----
    results = {
        "baseline": {
            "mean_per_drug_r": round(mean_base_r, 6),
            "global_r": round(global_r_base, 6),
            "n_drugs": len(common_drugs),
        },
        "with_moa_onehot": {
            "mean_per_drug_r": round(mean_oh_r, 6),
            "global_r": round(global_r_oh, 6),
            "n_drugs": len(common_drugs),
        },
        "delta_per_drug_r": round(delta_per_drug, 6),
        "delta_global_r": round(delta_global, 6),
        "n_folds": n_folds,
        "n_moa_classes": len(moa_classes),
        "conclusion": conclusion,
        "per_drug": per_drug_list,
    }

    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)
    out_path = report_data / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
