"""Part C: Extended drug representation ablation in OmniCancerV1 Transformer.

Tests LINCS, drug-target, and MoA one-hot under the same architecture as Part A,
to determine whether the drug-feature null generalizes beyond structural representations.

Conditions:
  lincs_pca64    — LINCS L1000 consensus signatures, PCA(64), 104 matched drugs only
  drug_target    — Binary drug-target vector from ChEMBL (~5145 targets), PCA(256)
  moa_onehot     — MoA one-hot (24 pathway classes)

Baseline from Part A:
  no_drug        — Zero vector (per-drug r ≈ 0.645)
  morgan_fp      — 2048-bit Morgan FP (per-drug r ≈ 0.652, Δ ≈ +0.008)

Usage:
    uv run python3 experiments/03_drug_feature_null/03_model_robustness/jobs/run_partC.py
    uv run python3 experiments/03_drug_feature_null/03_model_robustness/jobs/run_partC.py --fold 0
    uv run python3 experiments/03_drug_feature_null/03_model_robustness/jobs/run_partC.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).parents[4]
JOBS_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JOBS_DIR))

from src.models.transformer_encoder import TransformerEncoder as OmniCancerV1
from src.utils.paso_folds import load_cell_line_index, split_drug_blind_val
from src.utils.solutions import load_moa_annotations

import run_partA
from run_partA import (
    D_MODEL,
    DROPOUT,
    MODALITY_DROPOUT_P,
    N_HEADS,
    N_LAYERS,
    OMICS,
    build_full_dataset,
    load_paso_fold,
    map_fold_indices,
    train_fold_condition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K_FOLDS = 10
EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"

CONDITIONS = ["lincs_pca64", "drug_target", "moa_onehot"]
ALL_CONDITIONS = CONDITIONS + ["moa_permuted"]


# ---------------------------------------------------------------------------
# Drug feature loaders
# ---------------------------------------------------------------------------

def load_lincs_pca64(drug_to_idx: Dict[str, int]) -> Tuple[np.ndarray, set]:
    """Load LINCS signatures, PCA to 64-dim. Returns (matrix, matched_drug_names)."""
    idx_path = DATA_DIR / "lincs_drug_index.json"
    with open(idx_path) as f:
        lincs_idx = json.load(f)
    matched_names = lincs_idx["matched_drugs"]

    sig_path = DATA_DIR / "lincs_signatures.npy"
    raw_sigs = np.load(sig_path)  # (n_matched, 12127)

    pca = PCA(n_components=64, random_state=0)
    sigs_pca = pca.fit_transform(raw_sigs)  # (n_matched, 64)

    n_drugs = len(drug_to_idx)
    matrix = np.zeros((n_drugs, 64), dtype=np.float32)
    matched_set = set()
    for i, name in enumerate(matched_names):
        if name in drug_to_idx:
            matrix[drug_to_idx[name]] = sigs_pca[i]
            matched_set.add(name)

    logger.info("LINCS PCA-64: %d/%d drugs matched, explained_var=%.2f%%",
                len(matched_set), n_drugs, pca.explained_variance_ratio_.sum() * 100)
    return matrix, matched_set


def load_drug_target_pca(drug_to_idx: Dict[str, int], n_components: int = 256) -> np.ndarray:
    """Load drug-target binary features, PCA-reduce for Transformer input."""
    feat_path = DATA_DIR / "drug_target_features.npy"
    raw = np.load(feat_path)  # (n_drugs_in_file, n_targets)

    n_drugs = len(drug_to_idx)
    if raw.shape[0] >= n_drugs:
        features = raw[:n_drugs]
    else:
        features = np.zeros((n_drugs, raw.shape[1]), dtype=np.float32)
        features[:raw.shape[0]] = raw

    actual_dim = min(n_components, features.shape[1], features.shape[0])
    pca = PCA(n_components=actual_dim, random_state=0)
    reduced = pca.fit_transform(features)

    n_with_targets = int((features.sum(axis=1) > 0).sum())
    logger.info("Drug-target PCA-%d: %s → %s, %d/%d drugs have targets, explained_var=%.2f%%",
                actual_dim, features.shape, reduced.shape, n_with_targets, n_drugs,
                pca.explained_variance_ratio_.sum() * 100)
    return reduced.astype(np.float32)


def load_moa_onehot(drug_to_idx: Dict[str, int]) -> np.ndarray:
    """Load MoA one-hot from PASO annotations (Target Pathway)."""
    moa = load_moa_annotations()
    pathways = sorted(set(moa.values()))
    pathway_to_idx = {p: i for i, p in enumerate(pathways)}

    n_drugs = len(drug_to_idx)
    n_classes = len(pathways)
    matrix = np.zeros((n_drugs, n_classes), dtype=np.float32)

    for drug_name, drug_idx in drug_to_idx.items():
        pw = moa.get(drug_name)
        if pw and pw in pathway_to_idx:
            matrix[drug_idx, pathway_to_idx[pw]] = 1.0

    n_assigned = int((matrix.sum(1) > 0).sum())
    logger.info("MoA one-hot: %d classes, %d/%d drugs assigned, top classes: %s",
                n_classes, n_assigned, n_drugs, pathways[:5])
    return matrix


def load_moa_onehot_permuted(drug_to_idx: Dict[str, int], seed: int = 42) -> np.ndarray:
    """Load MoA one-hot with drug-pathway assignments randomly shuffled."""
    moa = load_moa_annotations()
    pathways = sorted(set(moa.values()))
    pathway_to_idx = {p: i for i, p in enumerate(pathways)}

    assigned_drugs = [(name, idx) for name, idx in drug_to_idx.items()
                      if moa.get(name) in pathway_to_idx]
    original_classes = [pathway_to_idx[moa[name]] for name, _ in assigned_drugs]

    rng = np.random.default_rng(seed)
    permuted_classes = original_classes.copy()
    rng.shuffle(permuted_classes)

    n_drugs = len(drug_to_idx)
    n_classes = len(pathways)
    matrix = np.zeros((n_drugs, n_classes), dtype=np.float32)

    for (_, drug_idx), cls_idx in zip(assigned_drugs, permuted_classes):
        matrix[drug_idx, cls_idx] = 1.0

    logger.info("MoA PERMUTED: %d classes, %d drugs shuffled (seed=%d)", n_classes, len(assigned_drugs), seed)
    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Part C: Extended Transformer drug-feature ablation")
    parser.add_argument("--fold", type=int, default=-1, help="Single fold (0-9). Omit for all.")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 3 epochs, fold 0 only.")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS,
                        choices=ALL_CONDITIONS, help="Which conditions to run.")
    args = parser.parse_args()

    global K_FOLDS
    run_single_fold: Optional[int] = args.fold if args.fold >= 0 else None
    conditions = args.conditions

    if args.smoke:
        run_partA.N_EPOCHS = 3
        K_FOLDS = 1
        if run_single_fold is None:
            run_single_fold = 0
        logger.info("SMOKE MODE: N_EPOCHS=3 K_FOLDS=1 fold=%d", run_single_fold)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Directories
    logs_dir = EXP_DIR / "logs"
    ckpt_dir = EXP_DIR / "checkpoints"
    report_dir = EXP_DIR / "report" / "data"
    for d in (logs_dir, ckpt_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_tag = f"fold{run_single_fold:02d}" if run_single_fold is not None else "all"
    fh = logging.FileHandler(logs_dir / f"run_partC_{log_tag}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("Part C: Extended drug representation ablation")
    logger.info("  Conditions: %s", conditions)
    logger.info("  Folds: %s", [run_single_fold] if run_single_fold is not None else list(range(K_FOLDS)))
    logger.info("  N_EPOCHS: %d", run_partA.N_EPOCHS)
    logger.info("=" * 70)

    # Load omics
    t0 = time.time()
    logger.info("Loading omics data...")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  Mutations: %s  (%.1fs)", rna.shape, mutations.shape, time.time() - t0)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    feature_dims = {"rna": rna.shape[1], "mutations": mutations.shape[1]}

    # Load PASO folds
    logger.info("Loading PASO 10-fold splits...")
    all_folds_raw = [load_paso_fold(k) for k in range(K_FOLDS)]
    all_drugs: set = set()
    for train_df, test_df in all_folds_raw:
        all_drugs |= set(train_df["drug"].unique()) | set(test_df["drug"].unique())
    all_drugs_sorted = sorted(all_drugs)
    drug_to_idx = {d: i for i, d in enumerate(all_drugs_sorted)}
    idx_to_drug = {i: d for d, i in drug_to_idx.items()}
    logger.info("Total drugs: %d", len(drug_to_idx))

    # Build full pair dataset
    logger.info("Building full pair dataset...")
    concat_np, cell_rows, drug_idxs_arr, targets, key_to_idx, row_to_depmap = (
        build_full_dataset(all_folds_raw, name_to_depmap, rna, mutations, drug_to_idx)
    )

    # Load drug feature matrices
    logger.info("Loading drug feature matrices...")
    drug_features: Dict[str, Tuple[np.ndarray, Optional[set]]] = {}

    if "lincs_pca64" in conditions:
        lincs_mat, lincs_matched = load_lincs_pca64(drug_to_idx)
        drug_features["lincs_pca64"] = (lincs_mat, lincs_matched)

    if "drug_target" in conditions:
        dt_mat = load_drug_target_pca(drug_to_idx)
        drug_features["drug_target"] = (dt_mat, None)

    if "moa_onehot" in conditions:
        moa_mat = load_moa_onehot(drug_to_idx)
        drug_features["moa_onehot"] = (moa_mat, None)

    if "moa_permuted" in conditions:
        moa_perm_mat = load_moa_onehot_permuted(drug_to_idx)
        drug_features["moa_permuted"] = (moa_perm_mat, None)

    logger.info("Drug feature dimensions: %s",
                {c: drug_features[c][0].shape[1] for c in conditions})

    # Run folds
    all_results: Dict[str, List[Dict]] = {c: [] for c in conditions}
    folds_to_run = [run_single_fold] if run_single_fold is not None else list(range(K_FOLDS))
    total_runs = len(folds_to_run) * len(conditions)
    run_count = 0

    for fold in folds_to_run:
        train_df, test_df = all_folds_raw[fold]
        full_train_idx = map_fold_indices(train_df, name_to_depmap, drug_to_idx, key_to_idx)
        test_idx = map_fold_indices(test_df, name_to_depmap, drug_to_idx, key_to_idx)
        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold)

        n_test_drugs = len(set(drug_idxs_arr[test_idx].tolist()))
        logger.info("=== Fold %d/%d  train=%d  val=%d  test=%d (%d drugs) ===",
                    fold, K_FOLDS - 1, len(train_idx), len(val_idx), len(test_idx), n_test_drugs)

        for condition in conditions:
            run_count += 1
            t_start = time.time()
            logger.info("--- [%d/%d] Fold %d  Condition: %s ---", run_count, total_runs, fold, condition)

            feat_matrix, matched_set = drug_features[condition]
            drug_fp_dim = feat_matrix.shape[1]

            # For LINCS: restrict test set to matched drugs only
            if matched_set is not None:
                test_idx_cond = np.array([
                    i for i in test_idx
                    if idx_to_drug[int(drug_idxs_arr[i])] in matched_set
                ], dtype=np.int64)
                n_test_drugs_cond = len(set(drug_idxs_arr[test_idx_cond].tolist()))
                logger.info("  LINCS restriction: %d → %d pairs (%d → %d drugs)",
                            len(test_idx), len(test_idx_cond), n_test_drugs, n_test_drugs_cond)
            else:
                test_idx_cond = test_idx

            torch.manual_seed(0)
            model = OmniCancerV1(
                feature_dims=feature_dims,
                modality_order=OMICS,
                drug_fp_dim=drug_fp_dim,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                dropout=DROPOUT,
                modality_dropout_p=MODALITY_DROPOUT_P,
            )
            n_params = sum(p.numel() for p in model.parameters())
            logger.info("  Model: drug_fp_dim=%d, total_params=%dK", drug_fp_dim, n_params // 1000)

            try:
                fold_result = train_fold_condition(
                    model=model,
                    concat_np=concat_np,
                    cell_rows=cell_rows,
                    drug_idxs_arr=drug_idxs_arr,
                    fp_matrix=feat_matrix,
                    targets=targets,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx_cond,
                    fold=fold,
                    condition=condition,
                    idx_to_drug=idx_to_drug,
                    row_to_depmap=row_to_depmap,
                    device=device,
                    ckpt_dir=ckpt_dir,
                    logs_dir=logs_dir,
                )
                elapsed = time.time() - t_start
                all_results[condition].append(fold_result)
                logger.info("  DONE: test_per_drug_r=%.4f  time=%.0fs",
                            fold_result["test_per_drug_r"], elapsed)
            except Exception as e:
                logger.error("[fold%d_%s] Error: %s", fold, condition, e, exc_info=True)
                raise
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()

    # Write shard or aggregate
    if run_single_fold is not None:
        shard = {"fold": run_single_fold, "results": {c: all_results[c] for c in conditions}}
        tag = "_permuted" if conditions == ["moa_permuted"] else ""
        shard_path = report_dir / f"fold_{run_single_fold:02d}_partC{tag}_results.json"
        with open(shard_path, "w") as f:
            json.dump(shard, f, indent=2)
        logger.info("Shard written: %s", shard_path)
        logger.info("Summary:")
        for c in conditions:
            for r in all_results[c]:
                logger.info("  %s  test_per_drug_r=%.4f", c, r["test_per_drug_r"])
        return

    # Aggregate all folds
    output: Dict[str, Any] = {}
    for condition in conditions:
        fold_rs = [r["test_per_drug_r"] for r in all_results[condition]]
        valid = [r for r in fold_rs if not np.isnan(r)]
        output[condition] = {
            "mean": float(np.mean(valid)) if valid else float("nan"),
            "std": float(np.std(valid)) if valid else float("nan"),
            "n_folds": len(valid),
            "folds": fold_rs,
        }

    # Part A baseline for delta computation
    parta_path = report_dir / "partA_metrics.json"
    if parta_path.exists():
        with open(parta_path) as f:
            parta = json.load(f)
        baseline = parta.get("no_drug", {}).get("mean", 0.645)
    else:
        baseline = 0.645
        logger.warning("partA_metrics.json not found, using default baseline=0.645")

    for condition in conditions:
        output[condition]["delta_vs_no_drug"] = output[condition]["mean"] - baseline

    metrics_path = report_dir / "partC_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 70)
    logger.info("FINAL RESULTS (Part C)")
    logger.info("-" * 70)
    logger.info("%-15s  %8s  %8s  %8s", "Condition", "Mean r", "Std", "Δ no_drug")
    logger.info("-" * 70)
    for condition in conditions:
        m = output[condition]
        logger.info("%-15s  %8.4f  %8.4f  %+8.4f", condition, m["mean"], m["std"], m["delta_vs_no_drug"])
    logger.info("-" * 70)
    logger.info("%-15s  %8.4f  %8s  %8s", "no_drug (ref)", baseline, "—", "—")
    logger.info("%-15s  %8.4f  %8s  %+8.4f", "morgan_fp (A)", baseline + 0.008, "—", "+0.008")
    logger.info("=" * 70)
    logger.info("Metrics written: %s", metrics_path)


if __name__ == "__main__":
    main()
