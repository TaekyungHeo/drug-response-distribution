"""Part A: 10-fold TransformerEncoder ablation — morgan_fp vs no_drug.

Usage:
    python experiments/03_drug_feature_null/03_model_robustness/jobs/run_partA.py
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

from src.data.drug_features import get_drug_fingerprints
from src.evaluation.per_drug import mean_per_drug_r
from src.models.transformer_encoder import TransformerEncoder
from src.utils.paso_folds import load_cell_line_index, split_drug_blind_val

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 256
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
MODALITY_DROPOUT_P = 0.3
# NOTE: PASO provides 10 drug-blind folds (each fold ~24 test drugs; 10 folds × ~23 = 233 total).
# The task spec says K_FOLDS=5, but folds 0-4 cover only ~118/233 test drugs.
# Part A uses all 10 folds to produce a complete 10-fold evaluation; this is consistent with
# how other experiments use PASO splits and with PASO's own protocol.
K_FOLDS = 10
OMICS = ["rna", "mutations"]

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

CONDITIONS = ["morgan_fp", "no_drug"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_paso_fold(fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{fold}.csv")
    test_df = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{fold}.csv")
    return train_df, test_df


def build_full_dataset(
    all_folds_raw: List[Tuple[pd.DataFrame, pd.DataFrame]],
    name_to_depmap: Dict[str, str],
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    drug_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
    """Build unified pair arrays from all folds.

    Returns:
        concat_np    — (n_cells, n_omics_features)
        cell_rows    — (n_pairs,) int32 index into concat_np
        drug_idxs    — (n_pairs,) int32
        targets      — (n_pairs,) float32
        key_to_idx   — (depmap_id, drug_name) → pair index
        cell_to_depmap — row_idx → depmap_id
    """
    all_pairs = pd.concat(
        [pd.concat([tr, te]) for tr, te in all_folds_raw]
    ).drop_duplicates()
    available_cells = set(rna.index) & set(mutations.index)

    valid_rows = []
    for _, row in all_pairs.iterrows():
        ccl = str(row["cell_line"]).upper()
        depmap = name_to_depmap.get(ccl)
        drug = row["drug"]
        if depmap and depmap in available_cells and drug in drug_to_idx:
            valid_rows.append({
                "depmap_id": depmap,
                "drug_name": drug,
                "ic50": float(str(row["IC50"])),
            })

    full_df = pd.DataFrame(valid_rows).drop_duplicates(subset=["depmap_id", "drug_name"])
    logger.info(
        "Full dataset: %d pairs, %d cells, %d drugs",
        len(full_df),
        full_df["depmap_id"].nunique(),
        full_df["drug_name"].nunique(),
    )

    all_cells = sorted(full_df["depmap_id"].unique())
    cell_to_row = {c: i for i, c in enumerate(all_cells)}
    row_to_depmap = {i: c for c, i in cell_to_row.items()}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    concat_np = np.concatenate([rna_arr, mut_arr], axis=1)

    cell_rows = np.array([cell_to_row[r] for r in full_df["depmap_id"]], dtype=np.int32)
    drug_idxs = np.array([drug_to_idx[d] for d in full_df["drug_name"]], dtype=np.int32)
    targets = full_df["ic50"].values.astype(np.float32)

    pair_keys = list(zip(full_df["depmap_id"], full_df["drug_name"]))
    key_to_idx = {k: i for i, k in enumerate(pair_keys)}

    return concat_np, cell_rows, drug_idxs, targets, key_to_idx, row_to_depmap


def map_fold_indices(
    df: pd.DataFrame,
    name_to_depmap: Dict[str, str],
    drug_to_idx: Dict[str, int],
    key_to_idx: Dict,
) -> np.ndarray:
    idx = []
    for _, row in df.iterrows():
        ccl = str(row["cell_line"]).upper()
        depmap = name_to_depmap.get(ccl)
        drug = row["drug"]
        if depmap and drug in drug_to_idx:
            k = (depmap, drug)
            if k in key_to_idx:
                idx.append(key_to_idx[k])
    return np.array(idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Prefetcher
# ---------------------------------------------------------------------------

class _Prefetcher:
    """Background thread that prepares batches and moves them to device."""

    def __init__(
        self,
        concat_np: np.ndarray,
        cell_rows: np.ndarray,
        drug_idxs: np.ndarray,
        fp_matrix: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        bs: int,
        device: torch.device,
    ) -> None:
        self._c = concat_np
        self._cr = cell_rows
        self._di = drug_idxs
        self._fp = fp_matrix
        self._t = targets
        self._idx = indices
        self._bs = bs
        self._dev = device
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._err: Optional[Exception] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            n = len(self._idx)
            perm = np.random.permutation(n)
            i = 0
            while not self._stop.is_set():
                if i + self._bs > n:
                    perm = np.random.permutation(n)
                    i = 0
                batch_idx = self._idx[perm[i : i + self._bs]]
                i += self._bs
                rows = self._cr[batch_idx]
                x = torch.from_numpy(self._c[rows].copy()).to(self._dev)
                fp = torch.from_numpy(self._fp[self._di[batch_idx]].copy()).to(self._dev)
                y = torch.from_numpy(self._t[batch_idx].copy()).to(self._dev)
                self._q.put((x, fp, y), timeout=60)
        except Exception as e:
            self._err = e

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._err:
            raise RuntimeError(f"Prefetcher error: {self._err}") from self._err
        return self._q.get(timeout=60)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_set_v1(
    model: TransformerEncoder,
    concat_np: np.ndarray,
    cell_rows: np.ndarray,
    drug_idxs: np.ndarray,
    fp_matrix: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    bs: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a set of indices. Returns (preds, targets, drug_idxs_out)."""
    model.eval()
    all_preds, all_tgts, all_drug_idxs = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), bs * 2):
            chunk = indices[start : start + bs * 2]
            rows = cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(device)
            fp = torch.from_numpy(fp_matrix[drug_idxs[chunk]].copy()).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model(x, fp)
            all_preds.append(pred.float().cpu().numpy())
            all_tgts.append(targets[chunk])
            all_drug_idxs.append(drug_idxs[chunk])
    return (
        np.concatenate(all_preds),
        np.concatenate(all_tgts),
        np.concatenate(all_drug_idxs),
    )


# ---------------------------------------------------------------------------
# Training loop for one fold × condition
# ---------------------------------------------------------------------------

def train_fold_condition(
    model: TransformerEncoder,
    concat_np: np.ndarray,
    cell_rows: np.ndarray,
    drug_idxs_arr: np.ndarray,
    fp_matrix: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    condition: str,
    idx_to_drug: Dict[int, str],
    row_to_depmap: Dict[int, str],
    device: torch.device,
    ckpt_dir: Path,
    logs_dir: Path,
) -> Dict[str, Any]:
    """Train one fold for one condition. Returns per-fold test metrics dict."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, N_EPOCHS - 10), eta_min=LR * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[10]
    )
    criterion = nn.MSELoss()

    steps_per_epoch = max(1, len(train_idx) // BATCH_SIZE)
    best_val_r = -np.inf
    best_state: Optional[Dict] = None
    best_epoch = 0

    fold_tag = f"fold{fold}_{condition}"
    epoch_rows: List[Dict[str, Any]] = []
    epoch_parquet = logs_dir / f"{fold_tag}_epoch_metrics.parquet"
    ckpt_path = ckpt_dir / f"{fold_tag}_best.pt"

    # --- sanity check tracking ---
    any_positive_train_r = False

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()

        # Reset GPU memory stats at epoch start
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        pf = _Prefetcher(
            concat_np, cell_rows, drug_idxs_arr, fp_matrix, targets,
            train_idx, BATCH_SIZE, device,
        )
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            x, fp, y = next(pf)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model(x, fp)
                loss = criterion(pred.float(), y)
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            )
            optimizer.step()
            epoch_loss += loss.item()
        pf.stop()
        scheduler.step()

        avg_train_loss = epoch_loss / steps_per_epoch
        lr_now = float(optimizer.param_groups[0]["lr"])

        # Evaluate
        train_preds, train_tgts, train_drug_idx_out = eval_set_v1(
            model, concat_np, cell_rows, drug_idxs_arr, fp_matrix, targets,
            train_idx, BATCH_SIZE, device,
        )
        val_preds, val_tgts, val_drug_idx_out = eval_set_v1(
            model, concat_np, cell_rows, drug_idxs_arr, fp_matrix, targets,
            val_idx, BATCH_SIZE, device,
        )

        train_drug_names = np.array([idx_to_drug[i] for i in train_drug_idx_out])
        val_drug_names = np.array([idx_to_drug[i] for i in val_drug_idx_out])

        train_per_drug_r = mean_per_drug_r(train_preds, train_tgts, train_drug_names)
        val_per_drug_r = mean_per_drug_r(val_preds, val_tgts, val_drug_names)

        # Val MSE loss
        val_loss = float(np.mean((val_preds - val_tgts) ** 2))

        # GPU memory
        gpu_peak_gb = 0.0
        gpu_reserved_gb = 0.0
        if device.type == "cuda":
            gpu_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_reserved_gb = torch.cuda.memory_reserved(device) / 1e9

        epoch_time = time.perf_counter() - t0

        # Checkpoint selection on val per-drug r
        if not np.isnan(val_per_drug_r) and val_per_drug_r > best_val_r:
            best_val_r = val_per_drug_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            torch.save(best_state, ckpt_path)

        # Sanity check after epoch 10
        if epoch == 10:
            if not np.isnan(train_per_drug_r) and train_per_drug_r > 0.0:
                any_positive_train_r = True
            else:
                warnings.warn(
                    f"[{fold_tag}] train_per_drug_r={train_per_drug_r:.4f} after epoch 10 "
                    f"— possible training failure. Check data or hyperparams.",
                    stacklevel=2,
                )
                logger.warning(
                    "[%s] SANITY FAIL: train_per_drug_r=%.4f at epoch 10", fold_tag, train_per_drug_r
                )

        if not np.isnan(train_per_drug_r) and train_per_drug_r > 0.0:
            any_positive_train_r = True

        epoch_row = {
            "fold": fold,
            "condition": condition,
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_loss),
            "train_per_drug_r": float(train_per_drug_r),
            "val_per_drug_r": float(val_per_drug_r),
            "learning_rate": lr_now,
            "grad_norm_pre_clip": float(grad_norm),
            "epoch_time_s": float(epoch_time),
            "gpu_memory_peak_gb": float(gpu_peak_gb),
            "gpu_memory_reserved_gb": float(gpu_reserved_gb),
        }
        epoch_rows.append(epoch_row)

        # Append-by-overwrite
        pd.DataFrame(epoch_rows).to_parquet(epoch_parquet, index=False)

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "[%s] ep %3d/%d  train_r=%.4f  val_r=%.4f  lr=%.2e  %.1fs",
                fold_tag, epoch, N_EPOCHS,
                train_per_drug_r, val_per_drug_r, lr_now, epoch_time,
            )

    # Final test evaluation at best checkpoint
    if best_state is None:
        logger.warning("[%s] No checkpoint saved — using final weights", fold_tag)
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_preds, test_tgts, test_drug_idx_out = eval_set_v1(
        model, concat_np, cell_rows, drug_idxs_arr, fp_matrix, targets,
        test_idx, BATCH_SIZE, device,
    )
    test_drug_names = np.array([idx_to_drug[i] for i in test_drug_idx_out])

    # Build cell_id array for test predictions
    test_cell_row_out = cell_rows[test_idx]
    test_cell_ids = np.array([row_to_depmap[int(r)] for r in test_cell_row_out])

    test_per_drug_r = mean_per_drug_r(test_preds, test_tgts, test_drug_names)

    logger.info(
        "[%s] DONE  best_epoch=%d  val_r=%.4f  test_per_drug_r=%.4f",
        fold_tag, best_epoch, best_val_r, test_per_drug_r,
    )

    # Save test predictions
    pred_df = pd.DataFrame({
        "drug_id": test_drug_names,
        "cell_id": test_cell_ids,
        "y_true": test_tgts.astype(np.float32),
        "y_pred": test_preds.astype(np.float32),
    })
    pred_path = logs_dir / f"{fold_tag}_test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    logger.info("[%s] Test predictions saved to %s", fold_tag, pred_path)

    if not any_positive_train_r:
        logger.warning(
            "[%s] WARNING: train_per_drug_r never exceeded 0.0 — "
            "possible complete training failure.",
            fold_tag,
        )

    return {
        "fold": fold,
        "condition": condition,
        "best_epoch": best_epoch,
        "best_val_r": float(best_val_r),
        "test_per_drug_r": float(test_per_drug_r),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int, default=-1,
        help="Run a single fold (0-based). Omit to run all K_FOLDS folds.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: 3 epochs, 1 fold only. For local correctness checks.",
    )
    args = parser.parse_args()
    run_single_fold: Optional[int] = args.fold if args.fold >= 0 else None

    if args.smoke:
        global N_EPOCHS, K_FOLDS  # noqa: PLW0603
        N_EPOCHS = 3
        K_FOLDS = 1
        if run_single_fold is None:
            run_single_fold = 0
        logger.info("SMOKE MODE: N_EPOCHS=%d K_FOLDS=%d fold=%d", N_EPOCHS, K_FOLDS, run_single_fold)

    # Device selection
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

    log_tag = f"fold{run_single_fold}" if run_single_fold is not None else "all"
    fh = logging.FileHandler(logs_dir / f"run_partA_{log_tag}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info(
        "Part A: 10-fold TransformerEncoder ablation | fold=%s",
        run_single_fold if run_single_fold is not None else "all",
    )

    # Load omics
    logger.info("Loading omics data...")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  Mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)

    feature_dims = {"rna": rna.shape[1], "mutations": mutations.shape[1]}
    logger.info("Feature dims: %s", feature_dims)

    # Load all 5 PASO folds
    logger.info("Loading PASO drug-blind folds 0–%d...", K_FOLDS - 1)
    all_folds_raw = []
    for k in range(K_FOLDS):
        train_df, test_df = load_paso_fold(k)
        all_folds_raw.append((train_df, test_df))

    # Build unified drug index from PASO folds (folds 0-4)
    all_drugs: set = set()
    for train_df, test_df in all_folds_raw:
        all_drugs |= set(train_df["drug"].unique()) | set(test_df["drug"].unique())
    all_drugs_sorted = sorted(all_drugs)
    drug_to_idx: Dict[str, int] = {d: i for i, d in enumerate(all_drugs_sorted)}
    idx_to_drug: Dict[int, str] = {i: d for d, i in drug_to_idx.items()}
    logger.info("Drugs in folds 0–4: %d", len(drug_to_idx))

    # Load Morgan fingerprints
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
    logger.info("FP matrix: %s", fp_matrix.shape)

    # Build full pair dataset
    concat_np, cell_rows, drug_idxs_arr, targets, key_to_idx, row_to_depmap = (
        build_full_dataset(all_folds_raw, name_to_depmap, rna, mutations, drug_to_idx)
    )

    # Zero drug fingerprint matrix for no_drug condition
    zero_fp_matrix = np.zeros_like(fp_matrix)

    # Run folds (all or single shard)
    all_results: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}
    folds_to_run = [run_single_fold] if run_single_fold is not None else list(range(K_FOLDS))
    logger.info("Running folds: %s", folds_to_run)

    for fold in folds_to_run:
        train_df, test_df = all_folds_raw[fold]

        full_train_idx = map_fold_indices(train_df, name_to_depmap, drug_to_idx, key_to_idx)
        test_idx = map_fold_indices(test_df, name_to_depmap, drug_to_idx, key_to_idx)
        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold)

        logger.info(
            "=== Fold %d  train=%d  val=%d  test=%d ===",
            fold, len(train_idx), len(val_idx), len(test_idx),
        )

        for condition in CONDITIONS:
            logger.info("--- Fold %d  Condition: %s ---", fold, condition)

            fp = fp_matrix if condition == "morgan_fp" else zero_fp_matrix

            torch.manual_seed(0)
            model = TransformerEncoder(
                feature_dims=feature_dims,
                modality_order=OMICS,
                drug_fp_dim=fp_matrix.shape[1],
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                dropout=DROPOUT,
                modality_dropout_p=MODALITY_DROPOUT_P,
            )

            try:
                fold_result = train_fold_condition(
                    model=model,
                    concat_np=concat_np,
                    cell_rows=cell_rows,
                    drug_idxs_arr=drug_idxs_arr,
                    fp_matrix=fp,
                    targets=targets,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    fold=fold,
                    condition=condition,
                    idx_to_drug=idx_to_drug,
                    row_to_depmap=row_to_depmap,
                    device=device,
                    ckpt_dir=ckpt_dir,
                    logs_dir=logs_dir,
                )
                all_results[condition].append(fold_result)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(
                    "[fold%d_%s] CUDA OOM: %s — try reducing BATCH_SIZE", fold, condition, e
                )
                raise
            except Exception as e:
                logger.error("[fold%d_%s] Unexpected error: %s", fold, condition, e, exc_info=True)
                raise
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()

    # If sharding: write per-fold shard and exit
    if run_single_fold is not None:
        shard: Dict[str, Any] = {
            "fold": run_single_fold,
            "results": {c: all_results[c] for c in CONDITIONS},
        }
        shard_path = report_dir / f"fold_{run_single_fold:02d}_partA_results.json"
        with open(shard_path, "w") as f:
            json.dump(shard, f, indent=2)
        logger.info("Shard written to %s", shard_path)
        for condition in CONDITIONS:
            for r in all_results[condition]:
                logger.info("  Fold %d  %s  test_per_drug_r=%.4f", run_single_fold, condition, r["test_per_drug_r"])
        return

    # Aggregate results and write partA_metrics.json
    output: Dict[str, Any] = {}
    for condition in CONDITIONS:
        fold_rs = [r["test_per_drug_r"] for r in all_results[condition]]
        valid = [r for r in fold_rs if not np.isnan(r)]
        output[condition] = {
            "mean": float(np.mean(valid)) if valid else float("nan"),
            "std": float(np.std(valid)) if valid else float("nan"),
            "folds": fold_rs,
        }

    delta = output["morgan_fp"]["mean"] - output["no_drug"]["mean"]
    output["delta_morgan_vs_no_drug"] = float(delta)

    metrics_path = report_dir / "partA_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("morgan_fp : mean=%.4f ± %.4f  %s",
                output["morgan_fp"]["mean"], output["morgan_fp"]["std"],
                " ".join(f"{r:.4f}" for r in output["morgan_fp"]["folds"]))
    logger.info("no_drug   : mean=%.4f ± %.4f  %s",
                output["no_drug"]["mean"], output["no_drug"]["std"],
                " ".join(f"{r:.4f}" for r in output["no_drug"]["folds"]))
    logger.info("delta_morgan_vs_no_drug: %.4f", delta)
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
