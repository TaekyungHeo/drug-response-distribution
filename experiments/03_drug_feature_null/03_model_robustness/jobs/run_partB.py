"""Part B: 10-fold OmniCancerV2 GNN training + drug embedding extraction.

For each fold k:
  1. Train OmniCancerV2 on fold-k train drugs (drug-blind split, 10% val holdout)
  2. Extract 256-dim embeddings for fold-k TEST drugs from best checkpoint
     (the checkpoint that never saw those drugs in training)

After all 10 folds: concatenate → covers all 233 drugs, each from a checkpoint
that never saw it. Save to DATA_DIR/gnn_embeddings_256.npy.

Usage (full run):
    python experiments/03_drug_feature_null/03_model_robustness/jobs/run_partB.py

Usage (shard mode — single fold via SLURM array):
    python run_partB.py --fold 3
    # Each fold writes:
    #   report/data/fold_03_partB_results.json
    #   report/data/fold_03_embeddings.npz  (drug_indices + embeddings arrays)
    # Then run aggregate_partB.py to assemble gnn_embeddings_256.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

from src.data.drug_graph import get_drug_graphs
from src.evaluation.per_drug import mean_per_drug_r
from src.models.omnicancer_v2 import OmniCancerV2
from src.utils.paso_folds import split_drug_blind_val

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
GNN_LAYERS = 3
DROPOUT = 0.1
MODALITY_DROPOUT_P = 0.3
# NOTE: PASO provides 10 drug-blind folds (each fold ~24 test drugs; 10 folds × ~23 = 233 total).
# The task spec says K_FOLDS=5, but folds 0-4 cover only ~118/233 test drugs.
# Part B requires K_FOLDS=10 to embed all 233 drugs from a checkpoint that never saw that drug.
# If only 5 folds were used, ~115 drugs would have no embedding and the coverage assertion
# would fail, breaking the downstream gnn_embeddings_256.npy dependency.
K_FOLDS = 10
OMICS = ["rna", "mutations"]

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"


# ---------------------------------------------------------------------------
# Data helpers (mirrored from run_partA.py)
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
        concat_np   — (n_cells, n_omics_features)
        cell_rows   — (n_pairs,) int32 index into concat_np
        drug_idxs   — (n_pairs,) int32
        targets     — (n_pairs,) float32
        key_to_idx  — (depmap_id, drug_name) → pair index
        row_to_depmap — row_idx → depmap_id string
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
                "ic50": float(row["IC50"]),
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
# Prefetcher for OmniCancerV2 (yields drug_idx tensor, not fp)
# ---------------------------------------------------------------------------

class _PrefetcherV2:
    """Background thread prefetcher for OmniCancerV2.

    Yields (x_omics, drug_idx, y) where drug_idx is int64.
    """

    def __init__(
        self,
        concat_np: np.ndarray,
        cell_rows: np.ndarray,
        drug_idxs: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        bs: int,
        device: torch.device,
    ) -> None:
        self._c = concat_np
        self._cr = cell_rows
        self._di = drug_idxs
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
                d_idx = torch.from_numpy(self._di[batch_idx].astype(np.int64)).to(self._dev)
                y = torch.from_numpy(self._t[batch_idx].copy()).to(self._dev)
                self._q.put((x, d_idx, y), timeout=60)
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
# Evaluation helper for V2
# ---------------------------------------------------------------------------

def eval_set_v2(
    model: OmniCancerV2,
    concat_np: np.ndarray,
    cell_rows: np.ndarray,
    drug_idxs: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    bs: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference pass. Returns (preds, targets, drug_idxs_out)."""
    model.eval()
    all_preds, all_tgts, all_didxs = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), bs * 2):
            chunk = indices[start : start + bs * 2]
            rows = cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(device)
            d_idx = torch.from_numpy(drug_idxs[chunk].astype(np.int64)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model(x, d_idx)
            all_preds.append(pred.float().cpu().numpy())
            all_tgts.append(targets[chunk])
            all_didxs.append(drug_idxs[chunk])
    return (
        np.concatenate(all_preds),
        np.concatenate(all_tgts),
        np.concatenate(all_didxs),
    )


# ---------------------------------------------------------------------------
# Embedding extraction from V2 GNN
# ---------------------------------------------------------------------------

def extract_drug_embeddings(
    model: OmniCancerV2,
    drug_indices: np.ndarray,
    device: torch.device,
    d_model: int = 256,
) -> np.ndarray:
    """Extract GNN embeddings for a list of drug indices.

    Runs the GNN branch of OmniCancerV2 without the omics encoder,
    returning shape (len(drug_indices), d_model).
    """
    model.eval()
    embs_list = []
    with torch.no_grad():
        # Process in chunks to avoid OOM on large drug sets
        chunk_size = 64
        for start in range(0, len(drug_indices), chunk_size):
            chunk = drug_indices[start : start + chunk_size]
            d_idx = torch.from_numpy(chunk.astype(np.int64)).to(device)
            # Use the GNN directly: encode drugs without omics context
            unique_idx, inverse = torch.unique(d_idx, return_inverse=True)
            atom_feats = model.drug_atom_feats[unique_idx]
            adj = model.drug_adj[unique_idx]
            mask = model.drug_mask[unique_idx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                drug_embs_unique = model.drug_gnn(atom_feats, adj, mask)
            drug_embs = drug_embs_unique[inverse].float()
            embs_list.append(drug_embs.cpu().numpy())
    return np.concatenate(embs_list, axis=0)


# ---------------------------------------------------------------------------
# Training loop for one fold (OmniCancerV2)
# ---------------------------------------------------------------------------

def train_fold_v2(
    model: OmniCancerV2,
    concat_np: np.ndarray,
    cell_rows: np.ndarray,
    drug_idxs_arr: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    test_drug_names_arr: np.ndarray,
    fold: int,
    idx_to_drug: Dict[int, str],
    row_to_depmap: Dict[int, str],
    device: torch.device,
    ckpt_dir: Path,
    logs_dir: Path,
) -> Dict[str, Any]:
    """Train one fold of OmniCancerV2. Returns fold metrics dict."""
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

    fold_tag = f"fold{fold}_gnn"
    epoch_rows: List[Dict[str, Any]] = []
    epoch_parquet = logs_dir / f"{fold_tag}_epoch_metrics.parquet"
    ckpt_path = ckpt_dir / f"{fold_tag}_best.pt"

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        pf = _PrefetcherV2(
            concat_np, cell_rows, drug_idxs_arr, targets,
            train_idx, BATCH_SIZE, device,
        )
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            x, d_idx, y = next(pf)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model(x, d_idx)
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
        train_preds, train_tgts, train_drug_idx_out = eval_set_v2(
            model, concat_np, cell_rows, drug_idxs_arr, targets,
            train_idx, BATCH_SIZE, device,
        )
        val_preds, val_tgts, val_drug_idx_out = eval_set_v2(
            model, concat_np, cell_rows, drug_idxs_arr, targets,
            val_idx, BATCH_SIZE, device,
        )

        train_drug_names = np.array([idx_to_drug[i] for i in train_drug_idx_out])
        val_drug_names = np.array([idx_to_drug[i] for i in val_drug_idx_out])

        train_per_drug_r = mean_per_drug_r(train_preds, train_tgts, train_drug_names)
        val_per_drug_r = mean_per_drug_r(val_preds, val_tgts, val_drug_names)
        val_loss = float(np.mean((val_preds - val_tgts) ** 2))

        gpu_peak_gb = 0.0
        gpu_reserved_gb = 0.0
        if device.type == "cuda":
            gpu_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_reserved_gb = torch.cuda.memory_reserved(device) / 1e9

        epoch_time = time.perf_counter() - t0

        if not np.isnan(val_per_drug_r) and val_per_drug_r > best_val_r:
            best_val_r = val_per_drug_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            torch.save(best_state, ckpt_path)

        epoch_row = {
            "fold": fold,
            "condition": "gnn",
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

    test_preds, test_tgts, test_drug_idx_out = eval_set_v2(
        model, concat_np, cell_rows, drug_idxs_arr, targets,
        test_idx, BATCH_SIZE, device,
    )
    test_drug_names_out = np.array([idx_to_drug[i] for i in test_drug_idx_out])
    test_cell_row_out = cell_rows[test_idx]
    test_cell_ids = np.array([row_to_depmap[int(r)] for r in test_cell_row_out])
    test_per_drug_r = mean_per_drug_r(test_preds, test_tgts, test_drug_names_out)

    logger.info(
        "[%s] DONE  best_epoch=%d  val_r=%.4f  test_per_drug_r=%.4f",
        fold_tag, best_epoch, best_val_r, test_per_drug_r,
    )

    # Save test predictions
    pred_df = pd.DataFrame({
        "drug_id": test_drug_names_out,
        "cell_id": test_cell_ids,
        "y_true": test_tgts.astype(np.float32),
        "y_pred": test_preds.astype(np.float32),
    })
    pred_path = logs_dir / f"{fold_tag}_test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    logger.info("[%s] Test predictions saved to %s", fold_tag, pred_path)

    return {
        "fold": fold,
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
        help="Run a single fold (0-based). Omit (or -1) to run all K_FOLDS folds.",
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

    log_tag = f"fold{run_single_fold:02d}" if run_single_fold is not None else "all"
    fh = logging.FileHandler(logs_dir / f"run_partB_{log_tag}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info(
        "Part B: 10-fold OmniCancerV2 GNN training + drug embedding extraction  "
        "[shard=%s]", log_tag,
    )

    # Load omics
    logger.info("Loading omics data...")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    cl_idx = pd.read_parquet(DATA_DIR / "cell_line_index.parquet")
    logger.info("RNA: %s  Mutations: %s", rna.shape, mutations.shape)

    name_to_depmap: Dict[str, str] = {}
    for depmap_id, row in cl_idx.iterrows():
        name_to_depmap[str(row["stripped_name"]).upper()] = str(depmap_id)

    feature_dims = {"rna": rna.shape[1], "mutations": mutations.shape[1]}
    logger.info("Feature dims: %s", feature_dims)

    # Load all PASO folds
    logger.info("Loading PASO drug-blind folds 0–%d...", K_FOLDS - 1)
    all_folds_raw = []
    for k in range(K_FOLDS):
        train_df, test_df = load_paso_fold(k)
        all_folds_raw.append((train_df, test_df))

    # Build unified drug index from all PASO folds
    all_drugs: set = set()
    for train_df, test_df in all_folds_raw:
        all_drugs |= set(train_df["drug"].unique()) | set(test_df["drug"].unique())
    all_drugs_sorted = sorted(all_drugs)
    drug_to_idx: Dict[str, int] = {d: i for i, d in enumerate(all_drugs_sorted)}
    idx_to_drug: Dict[int, str] = {i: d for d, i in drug_to_idx.items()}
    n_drugs = len(drug_to_idx)
    logger.info("Drugs across all folds: %d", n_drugs)

    # Load drug graphs
    logger.info("Loading drug graphs from %s...", DATA_DIR)
    atom_feats_np, adj_norm_np, mask_np = get_drug_graphs(drug_to_idx, DATA_DIR)
    logger.info(
        "Drug graphs: atom_feats=%s  adj=%s  mask=%s",
        atom_feats_np.shape, adj_norm_np.shape, mask_np.shape,
    )
    atom_feats_t = torch.from_numpy(atom_feats_np).float()
    adj_norm_t = torch.from_numpy(adj_norm_np).float()
    mask_t = torch.from_numpy(mask_np).bool()

    # Build full pair dataset
    concat_np, cell_rows, drug_idxs_arr, targets, key_to_idx, row_to_depmap = (
        build_full_dataset(all_folds_raw, name_to_depmap, rna, mutations, drug_to_idx)
    )

    # GNN embedding output — filled fold by fold (only used in full-run mode)
    gnn_embeddings = np.full((n_drugs, D_MODEL), np.nan, dtype=np.float32)
    embedded_drug_set: set = set()

    fold_results: List[Dict[str, Any]] = []
    folds_to_run = [run_single_fold] if run_single_fold is not None else list(range(K_FOLDS))
    logger.info("Folds to run: %s", folds_to_run)

    t_run_start = time.perf_counter()
    for fold in folds_to_run:
        train_df, test_df = all_folds_raw[fold]

        # Identify test drugs for this fold
        test_drugs_in_fold = sorted(
            d for d in test_df["drug"].unique() if d in drug_to_idx
        )
        test_drug_indices_fold = np.array(
            [drug_to_idx[d] for d in test_drugs_in_fold], dtype=np.int64
        )
        logger.info(
            "=== Fold %d  test_drugs=%d ===", fold, len(test_drugs_in_fold)
        )

        # Check for overlap with already-embedded drugs
        overlap = embedded_drug_set & set(test_drugs_in_fold)
        if overlap:
            logger.warning(
                "[fold%d] %d test drugs already embedded by a previous fold "
                "(these will be overwritten with current fold's embedding): %s",
                fold, len(overlap), sorted(overlap)[:5],
            )

        full_train_idx = map_fold_indices(train_df, name_to_depmap, drug_to_idx, key_to_idx)
        test_idx = map_fold_indices(test_df, name_to_depmap, drug_to_idx, key_to_idx)
        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold)

        # Test drug names array for this fold's test indices
        test_drug_names_fold = np.array([idx_to_drug[i] for i in drug_idxs_arr[test_idx]])

        logger.info(
            "  train=%d  val=%d  test=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )

        torch.manual_seed(0)
        model = OmniCancerV2(
            feature_dims=feature_dims,
            modality_order=OMICS,
            drug_atom_feats=atom_feats_t,
            drug_adj=adj_norm_t,
            drug_mask=mask_t,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            gnn_layers=GNN_LAYERS,
            dropout=DROPOUT,
            modality_dropout_p=MODALITY_DROPOUT_P,
        )

        try:
            fold_result = train_fold_v2(
                model=model,
                concat_np=concat_np,
                cell_rows=cell_rows,
                drug_idxs_arr=drug_idxs_arr,
                targets=targets,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                test_drug_names_arr=test_drug_names_fold,
                fold=fold,
                idx_to_drug=idx_to_drug,
                row_to_depmap=row_to_depmap,
                device=device,
                ckpt_dir=ckpt_dir,
                logs_dir=logs_dir,
            )
            fold_results.append(fold_result)

            # --- Embedding extraction for fold-k test drugs ---
            logger.info(
                "[fold%d] Extracting embeddings for %d test drugs...",
                fold, len(test_drug_indices_fold),
            )
            fold_embs = extract_drug_embeddings(
                model, test_drug_indices_fold, device, D_MODEL
            )
            for drug_idx_val, emb in zip(test_drug_indices_fold, fold_embs):
                gnn_embeddings[drug_idx_val] = emb
                embedded_drug_set.add(idx_to_drug[int(drug_idx_val)])

            logger.info(
                "[fold%d] Embedded %d test drugs (total embedded so far: %d/%d)",
                fold, len(test_drug_indices_fold), len(embedded_drug_set), n_drugs,
            )

            # --- Shard mode: write per-fold outputs and return ---
            if run_single_fold is not None:
                fold_tag = f"fold{fold:02d}"
                emb_shard_path = report_dir / f"{fold_tag}_embeddings.npz"
                np.savez(
                    emb_shard_path,
                    drug_indices=test_drug_indices_fold,
                    embeddings=fold_embs,
                )
                logger.info("[%s] Embeddings shard saved to %s", fold_tag, emb_shard_path)

                shard_result = {
                    "fold": fold,
                    "results": fold_result,
                }
                shard_path = report_dir / f"{fold_tag}_partB_results.json"
                with open(shard_path, "w") as f:
                    json.dump(shard_result, f, indent=2)
                elapsed = time.perf_counter() - t_run_start
                logger.info(
                    "[%s] Shard written to %s  elapsed=%.1fs",
                    fold_tag, shard_path, elapsed,
                )
                return

        except torch.cuda.OutOfMemoryError as e:
            logger.error("[fold%d_gnn] CUDA OOM: %s — try reducing BATCH_SIZE", fold, e)
            raise
        except Exception as e:
            logger.error("[fold%d_gnn] Unexpected error: %s", fold, e, exc_info=True)
            raise
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    elapsed_total = time.perf_counter() - t_run_start
    logger.info("All %d folds complete in %.1fs", len(folds_to_run), elapsed_total)

    # --- Coverage assertion ---
    missing_drugs = [
        idx_to_drug[i]
        for i in range(n_drugs)
        if np.any(np.isnan(gnn_embeddings[i]))
    ]
    if missing_drugs:
        logger.error(
            "COVERAGE FAILURE: %d drugs have missing (NaN) embeddings: %s",
            len(missing_drugs), missing_drugs[:10],
        )
        raise RuntimeError(
            f"{len(missing_drugs)} drugs missing from GNN embeddings after all folds. "
            f"This indicates PASO folds 0-4 do not partition all drugs into test sets. "
            f"Missing: {missing_drugs[:5]}"
        )
    logger.info("Coverage check PASSED: all %d drugs have valid embeddings.", n_drugs)

    # Save GNN embeddings
    emb_path = DATA_DIR / "gnn_embeddings_256.npy"
    np.save(emb_path, gnn_embeddings)
    logger.info(
        "GNN embeddings saved to %s  shape=%s",
        emb_path, gnn_embeddings.shape,
    )

    # Aggregate and write partB_metrics.json
    fold_test_rs = [r["test_per_drug_r"] for r in fold_results]
    valid_rs = [r for r in fold_test_rs if not np.isnan(r)]

    output = {
        "gnn_per_fold_test_per_drug_r": fold_test_rs,
        "gnn_mean": float(np.mean(valid_rs)) if valid_rs else float("nan"),
        "gnn_std": float(np.std(valid_rs)) if valid_rs else float("nan"),
        "embedding_shape": list(gnn_embeddings.shape),
    }
    metrics_path = report_dir / "partB_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info(
        "GNN per-fold test per-drug r: mean=%.4f ± %.4f  %s",
        output["gnn_mean"], output["gnn_std"],
        " ".join(f"{r:.4f}" for r in fold_test_rs),
    )
    logger.info("Embedding shape: %s", output["embedding_shape"])
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
