"""Phase 11: Reproduce PASO on their exact splits, quantify test-set snooping.

Two conditions:
  A) PASO protocol: 200 epochs, pick best TEST Pearson per fold (test-set snooping)
  B) Our protocol: 200 epochs, hold out val from train, pick best VAL epoch, report TEST

Both use PASO's pre-generated drug_blind fold CSVs (fixed splits).
Both use our TransformerEncoder model with Morgan FP + RNA+mutations.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate
from src.evaluation.per_drug import mean_per_drug_r
from src.models.transformer_encoder import TransformerEncoder
from src.training.trainer import _build_concat, _sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 512
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
MODALITY_DROPOUT_P = 0.3
OMICS = ["rna", "mutations"]
K_FOLDS = 10

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_DIR = ROOT / "external" / "PASO"
PASO_FOLDS_DIR = PASO_DIR / "data" / "10_fold_data" / "drug_blind"


def load_paso_fold(fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PASO's pre-generated train/test split for a fold."""
    train_df = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{fold}.csv")
    test_df = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{fold}.csv")
    return train_df, test_df


def build_fold_dataset(
    pairs_df: pd.DataFrame,
    name_to_depmap: Dict[str, str],
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    drug_to_idx: Dict[str, int],
    fp_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build arrays for a set of pairs. Returns concat_np, cell_rows, drug_idxs, targets."""
    available_cells = set(rna.index) & set(mutations.index)
    valid_rows = []
    for _, row in pairs_df.iterrows():
        ccl = str(row["cell_line"]).upper()
        depmap = name_to_depmap.get(ccl)
        drug = row["drug"]
        if depmap and depmap in available_cells and drug in drug_to_idx:
            valid_rows.append({"depmap_id": depmap, "drug_name": drug, "ic50": float(row["IC50"])})

    df = pd.DataFrame(valid_rows)
    all_cells = sorted(df["depmap_id"].unique())
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    concat_np = np.concatenate([rna_arr, mut_arr], axis=1)

    cell_rows = np.array([cell_to_row[r] for r in df["depmap_id"]], dtype=np.int32)
    drug_idxs = np.array([drug_to_idx[d] for d in df["drug_name"]], dtype=np.int32)
    targets = df["ic50"].values.astype(np.float32)

    return concat_np, cell_rows, drug_idxs, targets


class _Prefetcher:
    import queue
    import threading

    def __init__(self, concat_np, cell_rows, drug_idxs, fp_matrix, targets, indices, bs, device):
        import queue, threading
        self._c, self._cr, self._di, self._fp = concat_np, cell_rows, drug_idxs, fp_matrix
        self._t, self._idx, self._bs, self._dev = targets, indices, bs, device
        self._q = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._err = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            n = len(self._idx)
            perm = np.random.permutation(n)
            i = 0
            while not self._stop.is_set():
                if i + self._bs > n:
                    perm = np.random.permutation(n)
                    i = 0
                idx = self._idx[perm[i:i+self._bs]]
                i += self._bs
                rows = self._cr[idx]
                x = torch.from_numpy(self._c[rows].copy()).to(self._dev)
                fp = torch.from_numpy(self._fp[self._di[idx]].copy()).to(self._dev)
                y = torch.from_numpy(self._t[idx].copy()).to(self._dev)
                self._q.put((x, fp, y), timeout=60)
        except Exception as e:
            self._err = e

    def __next__(self):
        if self._err: raise RuntimeError(str(self._err))
        return self._q.get(timeout=60)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


def split_drug_blind_val(
    drug_idxs_arr: np.ndarray,
    full_train_idx: np.ndarray,
    fold_i: int,
    val_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split training indices into train/val by holding out a fraction of drugs.

    Val drugs are drawn entirely from the training set (no test leakage).
    Because val drugs never appear in training, val_r and test_r are on the
    same drug-blind distribution — a prerequisite for fair snooping measurement.

    Returns:
        (train_idx, val_idx) — both are subsets of full_train_idx
    """
    train_drug_indices = np.unique(drug_idxs_arr[full_train_idx])
    rng = np.random.default_rng(42 + fold_i)
    shuffled = rng.permutation(len(train_drug_indices))
    n_val_drugs = max(1, int(len(train_drug_indices) * val_frac))
    val_drug_set = set(train_drug_indices[shuffled[:n_val_drugs]].tolist())
    val_mask = np.isin(drug_idxs_arr[full_train_idx], list(val_drug_set))
    return full_train_idx[~val_mask], full_train_idx[val_mask]


def eval_set(model, concat_np, cell_rows, drug_idxs, fp_matrix, targets, indices, bs, device):
    model.eval()
    preds, tgts, drug_idx_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(indices), bs * 2):
            chunk = indices[i:i+bs*2]
            rows = cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(device)
            fp = torch.from_numpy(fp_matrix[drug_idxs[chunk]].copy()).to(device)
            pred = model(x, fp)
            preds.append(pred.cpu().numpy())
            tgts.append(targets[chunk])
            drug_idx_list.append(drug_idxs[chunk])
    return np.concatenate(preds), np.concatenate(tgts), np.concatenate(drug_idx_list)


def train_fold(
    model, concat_np, cell_rows, drug_idxs, fp_matrix, targets,
    train_idx, val_idx, test_idx,
    n_epochs, bs, lr, device, fold_name,
    idx_to_drug: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Train one fold. Track best-test (PASO) and best-val (ours) separately."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - 10), eta_min=lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])
    criterion = nn.MSELoss()

    steps = len(train_idx) // bs
    best_val_r, best_val_state = -np.inf, None
    best_test_r, best_test_state, best_test_epoch = -np.inf, None, 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        pf = _Prefetcher(concat_np, cell_rows, drug_idxs, fp_matrix, targets, train_idx, bs, device)
        for _ in range(steps):
            x, fp, y = next(pf)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, fp)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        pf.stop()
        scheduler.step()

        # Evaluate on val and test
        val_preds, val_tgts, _ = eval_set(model, concat_np, cell_rows, drug_idxs, fp_matrix, targets, val_idx, bs, device)
        val_r = float(evaluate(val_tgts, val_preds)["pearson_r"])

        test_preds, test_tgts, _ = eval_set(model, concat_np, cell_rows, drug_idxs, fp_matrix, targets, test_idx, bs, device)
        test_r = float(evaluate(test_tgts, test_preds)["pearson_r"])

        if val_r > best_val_r:
            best_val_r = val_r
            best_val_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_test_r = test_r
            best_val_epoch = epoch

        if test_r > best_test_r:
            best_test_r = test_r
            best_test_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_test_epoch = epoch

        elapsed = time.perf_counter() - t0
        if epoch % 20 == 0 or epoch == 1:
            logger.info("[%s] ep %3d/%d  val_r=%.4f  test_r=%.4f  %.1fs",
                        fold_name, epoch, n_epochs, val_r, test_r, elapsed)

    # Final: evaluate with best-val model (fair protocol)
    model.load_state_dict(best_val_state)
    final_preds, final_tgts, final_drug_idxs = eval_set(
        model, concat_np, cell_rows, drug_idxs, fp_matrix, targets, test_idx, bs, device)
    final_r = float(evaluate(final_tgts, final_preds)["pearson_r"])
    fair_per_drug_r = None
    if idx_to_drug is not None:
        final_drug_names = np.array([idx_to_drug[i] for i in final_drug_idxs])
        fair_per_drug_r = float(mean_per_drug_r(final_preds, final_tgts, final_drug_names))

    # Also evaluate with best-test model (PASO-style snooping)
    model.load_state_dict(best_test_state)
    snooping_preds, snooping_tgts, snooping_drug_idxs = eval_set(
        model, concat_np, cell_rows, drug_idxs, fp_matrix, targets, test_idx, bs, device)
    paso_style_per_drug_r = None
    if idx_to_drug is not None:
        snooping_drug_names = np.array([idx_to_drug[i] for i in snooping_drug_idxs])
        paso_style_per_drug_r = float(mean_per_drug_r(snooping_preds, snooping_tgts, snooping_drug_names))

    logger.info(
        "  %s | best_test=%.4f (ep%d, PASO-style) per_drug=%.4f  "
        "best_val_test=%.4f (ep%d, ours) per_drug=%.4f",
        fold_name, best_test_r, best_test_epoch, paso_style_per_drug_r or float("nan"),
        final_r, best_val_epoch, fair_per_drug_r or float("nan"),
    )

    return {
        "best_test_r": best_test_r,
        "best_test_epoch": best_test_epoch,
        "best_val_test_r": final_r,
        "best_val_epoch": best_val_epoch,
        "best_val_r": best_val_r,
        "fair_per_drug_r": fair_per_drug_r,
        "paso_style_per_drug_r": paso_style_per_drug_r,
    }


def main() -> None:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXP_DIR / "results" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(EXP_DIR / "logs" / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("Phase 11 PASO Reproduction | run_dir=%s", run_dir)

    # Load omics
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    cl_idx = pd.read_parquet(DATA_DIR / "cell_line_index.parquet")

    name_to_depmap: Dict[str, str] = {}
    for depmap_id, row in cl_idx.iterrows():
        name_to_depmap[row["stripped_name"].upper()] = str(depmap_id)

    feature_dims = {"rna": rna.shape[1], "mutations": mutations.shape[1]}

    # Load all 10 folds and get full drug set
    all_folds_raw = []
    for fold_i in range(K_FOLDS):
        train_df, test_df = load_paso_fold(fold_i)
        all_folds_raw.append((train_df, test_df))

    # Build drug index from all folds
    all_drugs = set()
    for train_df, test_df in all_folds_raw:
        all_drugs |= set(train_df["drug"].unique()) | set(test_df["drug"].unique())
    all_drugs = sorted(all_drugs)
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    idx_to_drug = {i: d for d, i in drug_to_idx.items()}

    # Get Morgan FP for all drugs
    from src.data.drug_features import get_drug_fingerprints
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
    logger.info("Drugs: %d, FP shape: %s", len(drug_to_idx), fp_matrix.shape)

    # Build full dataset (all cells across all folds)
    all_pairs = pd.concat([pd.concat([tr, te]) for tr, te in all_folds_raw]).drop_duplicates()
    available_cells = set(rna.index) & set(mutations.index)

    valid_rows = []
    for _, row in all_pairs.iterrows():
        ccl = str(row["cell_line"]).upper()
        depmap = name_to_depmap.get(ccl)
        drug = row["drug"]
        if depmap and depmap in available_cells and drug in drug_to_idx:
            valid_rows.append({"depmap_id": depmap, "drug_name": drug, "ic50": float(row["IC50"])})
    full_df = pd.DataFrame(valid_rows)
    logger.info("Full dataset: %d pairs, %d cells, %d drugs",
                len(full_df), full_df["depmap_id"].nunique(), full_df["drug_name"].nunique())

    all_cells = sorted(full_df["depmap_id"].unique())
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    concat_np = np.concatenate([rna_arr, mut_arr], axis=1)

    cell_rows = np.array([cell_to_row[r] for r in full_df["depmap_id"]], dtype=np.int32)
    drug_idxs_arr = np.array([drug_to_idx[d] for d in full_df["drug_name"]], dtype=np.int32)
    targets = full_df["ic50"].values.astype(np.float32)

    # Build pair key → index mapping
    pair_keys = list(zip(full_df["depmap_id"], full_df["drug_name"]))
    key_to_idx = {k: i for i, k in enumerate(pair_keys)}

    # Run 10-fold CV on PASO's exact splits
    results: Dict[str, Any] = {"folds": []}
    results_path = run_dir / "results.json"

    for fold_i in range(K_FOLDS):
        train_df, test_df = all_folds_raw[fold_i]
        logger.info("=== Fold %d/%d ===", fold_i + 1, K_FOLDS)

        # Map PASO fold rows to our indices
        def map_indices(df):
            idx = []
            for _, row in df.iterrows():
                ccl = str(row["cell_line"]).upper()
                depmap = name_to_depmap.get(ccl)
                drug = row["drug"]
                if depmap and (depmap, drug) in key_to_idx:
                    idx.append(key_to_idx[(depmap, drug)])
            return np.array(idx, dtype=np.int64)

        full_train_idx = map_indices(train_df)
        test_idx = map_indices(test_df)

        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i)

        logger.info("  train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

        model = TransformerEncoder(
            feature_dims=feature_dims, modality_order=OMICS,
            drug_fp_dim=fp_matrix.shape[1],
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            dropout=DROPOUT, modality_dropout_p=MODALITY_DROPOUT_P,
        )

        fold_result = train_fold(
            model, concat_np, cell_rows, drug_idxs_arr, fp_matrix, targets,
            train_idx, val_idx, test_idx,
            n_epochs=N_EPOCHS, bs=BATCH_SIZE, lr=LR, device=device,
            fold_name=f"fold{fold_i}", idx_to_drug=idx_to_drug,
        )
        results["folds"].append(fold_result)

        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

        del model
        if device == "mps":
            torch.mps.empty_cache()

    # Summary
    paso_style = [f["best_test_r"] for f in results["folds"]]
    our_style = [f["best_val_test_r"] for f in results["folds"]]
    results["paso_style_mean"] = float(np.mean(paso_style))
    results["paso_style_std"] = float(np.std(paso_style))
    results["our_style_mean"] = float(np.mean(our_style))
    results["our_style_std"] = float(np.std(our_style))

    paso_per_drug = [f["paso_style_per_drug_r"] for f in results["folds"] if f.get("paso_style_per_drug_r") is not None]
    fair_per_drug = [f["fair_per_drug_r"] for f in results["folds"] if f.get("fair_per_drug_r") is not None]
    if paso_per_drug:
        results["paso_style_per_drug_mean"] = float(np.mean(paso_per_drug))
        results["paso_style_per_drug_std"] = float(np.std(paso_per_drug))
    if fair_per_drug:
        results["fair_per_drug_mean"] = float(np.mean(fair_per_drug))
        results["fair_per_drug_std"] = float(np.std(fair_per_drug))

    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("PASO-style (best test):  mean=%.4f ± %.4f  %s",
                np.mean(paso_style), np.std(paso_style),
                " ".join(f"{r:.3f}" for r in paso_style))
    logger.info("Our-style (best val):    mean=%.4f ± %.4f  %s",
                np.mean(our_style), np.std(our_style),
                " ".join(f"{r:.3f}" for r in our_style))
    logger.info("Snooping inflation:      %.4f", np.mean(paso_style) - np.mean(our_style))
    if paso_per_drug:
        logger.info("PASO-style per-drug r:   mean=%.4f ± %.4f", np.mean(paso_per_drug), np.std(paso_per_drug))
    if fair_per_drug:
        logger.info("Fair per-drug r:         mean=%.4f ± %.4f", np.mean(fair_per_drug), np.std(fair_per_drug))

    # Write metadata.json for template rendering
    import shutil
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_hash = "unknown"
    metadata = {
        "experiment": "paso_decomposition",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "git_hash": git_hash,
        "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "TransformerEncoder",
        "drug_features": "Morgan FP (2048-bit)",
        "omics": OMICS,
        "paso_splits": "external/PASO/data/10_fold_data/drug_blind/",
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "total_pairs": len(full_df),
        "cells": int(full_df["depmap_id"].nunique()),
        "drugs": int(full_df["drug_name"].nunique()),
    }
    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)
    with open(report_data / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    shutil.copy(results_path, report_data / "results.json")
    logger.info("Metadata written to %s", report_data / "metadata.json")


if __name__ == "__main__":
    main()