"""Experiment 06: Objective axis — MSE vs RankNet on MLP.

Three conditions:
  mlp_mse_no_drug    — MSE, cell features only (750-dim PCA)
  mlp_mse_morgan     — MSE, cell + Morgan FP (2798-dim)
  mlp_ranknet_morgan — RankNet BCE, cell + Morgan FP (2798-dim)

Cell features: RNA PCA(550) + mutation PCA(200) via compress_cell (fit on train cells only).
Splits: PASO 10-fold drug-blind CV with 10% drug-blind val holdout per fold.
Checkpoint selection: best val per-drug Pearson r.
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

from src.data.drug_features import get_drug_fingerprints
from src.evaluation.per_drug import mean_per_drug_r, per_drug_r
from src.utils.paso_folds import build_pair_index, load_paso_folds, map_fold_indices, split_drug_blind_val
from src.utils.ridge import compress_cell

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 256
DROPOUT = 0.1
K_FOLDS = 10
MODEL_SEED = 0
N_PAIRS_PER_DRUG = 100

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

CONDITIONS = ["mlp_mse_no_drug", "mlp_mse_morgan", "mlp_ranknet_morgan"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MSE prefetcher
# ---------------------------------------------------------------------------
class _MSEPrefetcher:
    """Background thread that shuffles and batches tensors for MSE training."""

    def __init__(
        self,
        cell_features: np.ndarray,   # (all_cells, cell_dim), already PCA-compressed
        cell_rows: np.ndarray,        # (N,) int32 index into cell_features
        fp_matrix: Optional[np.ndarray],  # (n_drugs, fp_dim) or None for no_drug
        drug_idxs: np.ndarray,        # (N,) int32
        targets: np.ndarray,          # (N,) float32
        indices: np.ndarray,          # subset of [0, N)
        bs: int,
        device: str,
    ):
        import queue
        import threading

        self._cf = cell_features
        self._cr = cell_rows
        self._fp = fp_matrix
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
                cell_t = torch.from_numpy(self._cf[rows].copy()).to(self._dev)
                y_t = torch.from_numpy(self._t[batch_idx].copy()).to(self._dev)

                if self._fp is not None:
                    fp_t = torch.from_numpy(self._fp[self._di[batch_idx]].copy()).to(self._dev)
                    x_t = torch.cat([cell_t, fp_t], dim=-1)
                else:
                    x_t = cell_t

                self._q.put((x_t, y_t), timeout=60)
        except Exception as exc:
            self._err = exc

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._err:
            raise RuntimeError(str(self._err))
        return self._q.get(timeout=60)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Evaluation helper (no_drug and morgan both use this)
# ---------------------------------------------------------------------------
def eval_set(
    model: SimpleMLP,
    cell_features: np.ndarray,
    cell_rows: np.ndarray,
    fp_matrix: Optional[np.ndarray],
    drug_idxs: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    bs: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, tgts, drug_idxs_out) for a set of sample indices."""
    model.eval()
    preds_list, tgts_list, didx_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(indices), bs * 2):
            chunk = indices[i : i + bs * 2]
            rows = cell_rows[chunk]
            cell_t = torch.from_numpy(cell_features[rows].copy()).to(device)
            if fp_matrix is not None:
                fp_t = torch.from_numpy(fp_matrix[drug_idxs[chunk]].copy()).to(device)
                x_t = torch.cat([cell_t, fp_t], dim=-1)
            else:
                x_t = cell_t
            pred = model(x_t)
            preds_list.append(pred.cpu().numpy())
            tgts_list.append(targets[chunk])
            didx_list.append(drug_idxs[chunk])
    return (
        np.concatenate(preds_list),
        np.concatenate(tgts_list),
        np.concatenate(didx_list),
    )


# ---------------------------------------------------------------------------
# RankNet pair sampling (streaming)
# ---------------------------------------------------------------------------
def sample_ranknet_pairs(
    train_idx: np.ndarray,
    drug_idxs: np.ndarray,
    n_pairs_per_drug: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Sample pairs for RankNet within each drug group.

    Returns:
        idx_i, idx_j  — indices into the full dataset arrays (not train_idx positions)
        n_drugs_with_pairs — count of drugs that contributed ≥1 pair
        total_pairs — total number of sampled pairs
    """
    # Group train_idx by drug
    drug_groups: Dict[int, List[int]] = defaultdict(list)
    for pos in train_idx:
        drug_groups[int(drug_idxs[pos])].append(int(pos))

    idx_i_list: List[int] = []
    idx_j_list: List[int] = []
    n_drugs_with_pairs = 0

    for d_members in drug_groups.values():
        n = len(d_members)
        if n < 2:
            continue
        n_drugs_with_pairs += 1
        arr = np.array(d_members)
        # Maximum possible pairs
        max_pairs = n * (n - 1) // 2
        n_sample = min(n_pairs_per_drug, max_pairs)
        # Sample pairs without replacement via two random permutations of indices
        # (fast and avoids building the full pair matrix)
        sampled_pairs: set = set()
        attempts = 0
        max_attempts = n_sample * 8
        while len(sampled_pairs) < n_sample and attempts < max_attempts:
            need = n_sample - len(sampled_pairs)
            a_picks = rng.integers(0, n, size=need)
            b_picks = rng.integers(0, n - 1, size=need)
            # Shift b to avoid equal indices
            b_picks[b_picks >= a_picks] += 1
            for a, b in zip(a_picks.tolist(), b_picks.tolist()):
                sampled_pairs.add((int(a), int(b)) if a < b else (int(b), int(a)))
                if len(sampled_pairs) >= n_sample:
                    break
            attempts += need

        for a, b in sampled_pairs:
            idx_i_list.append(int(arr[a]))
            idx_j_list.append(int(arr[b]))

    return (
        np.array(idx_i_list, dtype=np.int64),
        np.array(idx_j_list, dtype=np.int64),
        n_drugs_with_pairs,
        len(idx_i_list),
    )


# ---------------------------------------------------------------------------
# Training: one epoch (MSE)
# ---------------------------------------------------------------------------
def train_epoch_mse(
    model: SimpleMLP,
    optimizer: torch.optim.Optimizer,
    cell_features: np.ndarray,
    cell_rows: np.ndarray,
    fp_matrix: Optional[np.ndarray],
    drug_idxs: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    bs: int,
    device: str,
    use_amp: bool,
) -> Tuple[float, float]:
    """Run one MSE training epoch. Returns (mean_train_loss, grad_norm_pre_clip)."""
    model.train()
    criterion = nn.MSELoss()
    steps = max(1, len(train_idx) // bs)

    pf = _MSEPrefetcher(
        cell_features, cell_rows, fp_matrix, drug_idxs, targets, train_idx, bs, device
    )
    total_loss = 0.0
    total_grad_norm = 0.0

    for _ in range(steps):
        x, y = next(pf)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
                loss = criterion(pred, y)
        else:
            pred = model(x)
            loss = criterion(pred, y)

        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        )
        optimizer.step()
        total_loss += loss.item()
        total_grad_norm += grad_norm

    pf.stop()
    return total_loss / steps, total_grad_norm / steps


# ---------------------------------------------------------------------------
# Training: one epoch (RankNet)
# ---------------------------------------------------------------------------
def train_epoch_ranknet(
    model: SimpleMLP,
    optimizer: torch.optim.Optimizer,
    cell_features: np.ndarray,
    cell_rows: np.ndarray,
    fp_matrix: np.ndarray,   # required for RankNet (morgan condition)
    drug_idxs: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    bs: int,
    device: str,
    use_amp: bool,
    epoch: int,
    n_train_drugs: int,
) -> Tuple[float, float, float, float, float]:
    """Run one RankNet epoch.

    Returns:
        (mean_loss, grad_norm_pre_clip, pairs_per_batch_mean,
         drugs_contributing_pairs_fraction, pair_label_positive_fraction)
    """
    model.train()
    bce = nn.BCEWithLogitsLoss()
    steps = max(1, len(train_idx) // bs)
    rng = np.random.default_rng(seed=epoch * 10007)

    total_loss = 0.0
    total_grad_norm = 0.0
    total_pairs = 0
    total_drugs_contributing = 0
    positive_labels = 0
    total_labels = 0

    for step in range(steps):
        # Sample all train drugs each step for N_PAIRS_PER_DRUG pairs each
        idx_i, idx_j, n_drugs_with_pairs, n_pairs = sample_ranknet_pairs(
            train_idx, drug_idxs, N_PAIRS_PER_DRUG, rng
        )
        if n_pairs == 0:
            continue

        # Fraction of drugs contributing (over all train drugs)
        total_drugs_contributing += n_drugs_with_pairs
        total_pairs += n_pairs

        # Build input tensors for pairs
        rows_i = cell_rows[idx_i]
        rows_j = cell_rows[idx_j]
        cell_i = torch.from_numpy(cell_features[rows_i].copy()).to(device)
        cell_j = torch.from_numpy(cell_features[rows_j].copy()).to(device)
        fp_i = torch.from_numpy(fp_matrix[drug_idxs[idx_i]].copy()).to(device)
        fp_j = torch.from_numpy(fp_matrix[drug_idxs[idx_j]].copy()).to(device)
        x_i = torch.cat([cell_i, fp_i], dim=-1)
        x_j = torch.cat([cell_j, fp_j], dim=-1)

        y_i_val = targets[idx_i]
        y_j_val = targets[idx_j]
        # label = 1 if target_i > target_j else 0 (strict, no tie credit)
        labels = (y_i_val > y_j_val).astype(np.float32)
        labels_t = torch.from_numpy(labels).to(device)
        positive_labels += int(labels.sum())
        total_labels += len(labels)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                score_i = model(x_i)
                score_j = model(x_j)
                logits = score_i - score_j
                loss = bce(logits, labels_t)
        else:
            score_i = model(x_i)
            score_j = model(x_j)
            logits = score_i - score_j
            loss = bce(logits, labels_t)

        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        )
        optimizer.step()
        total_loss += loss.item()
        total_grad_norm += grad_norm

    mean_loss = total_loss / steps
    mean_grad = total_grad_norm / steps
    mean_pairs = total_pairs / steps
    # drugs_contributing: mean fraction of train drugs per step with ≥1 pair
    drugs_frac = total_drugs_contributing / (steps * n_train_drugs) if n_train_drugs > 0 else 0.0
    pos_frac = positive_labels / total_labels if total_labels > 0 else 0.0

    return mean_loss, mean_grad, mean_pairs, drugs_frac, pos_frac


# ---------------------------------------------------------------------------
# Per-fold training loop
# ---------------------------------------------------------------------------
def train_condition(
    condition: str,
    cell_features: np.ndarray,        # (all_cells, cell_dim) — PCA-compressed, fold-specific
    cell_rows: np.ndarray,             # (N,) int32 index into cell_features
    fp_matrix: Optional[np.ndarray],  # (n_drugs, 2048) or None
    drug_idxs: np.ndarray,            # (N,) int32
    targets: np.ndarray,              # (N,) float32
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    idx_to_drug: Dict[int, str],
    idx_to_cell: Dict[int, str],      # dataset row → depmap_id
    fold: int,
    device: str,
) -> Dict[str, Any]:
    """Train one condition for one fold. Returns fold results dict."""
    use_drug = condition in ("mlp_mse_morgan", "mlp_ranknet_morgan")
    is_ranknet = condition == "mlp_ranknet_morgan"

    in_dim = cell_features.shape[1]
    if use_drug and fp_matrix is not None:
        in_dim += fp_matrix.shape[1]

    torch.manual_seed(MODEL_SEED)
    model = SimpleMLP(in_dim=in_dim, hidden_dim=512, dropout=DROPOUT).to(device)

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

    use_amp = device == "cuda"
    fp_for_train = fp_matrix if use_drug else None

    # Pre-compute drug_to_train_indices for RankNet (outside epoch loop)
    n_train_drugs = len(np.unique(drug_idxs[train_idx])) if is_ranknet else 0

    best_val_r = -np.inf
    best_val_state: Optional[Dict] = None
    best_val_epoch = 0

    epoch_rows: List[Dict[str, Any]] = []

    logs_dir = EXP_DIR / "logs"
    ckpt_dir = EXP_DIR / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()

        # ----- train step -----
        if is_ranknet:
            (
                train_loss,
                grad_norm,
                pairs_per_batch_mean,
                drugs_contributing_frac,
                pair_label_pos_frac,
            ) = train_epoch_ranknet(
                model, optimizer, cell_features, cell_rows, fp_matrix,  # type: ignore[arg-type]
                drug_idxs, targets, train_idx,
                bs=BATCH_SIZE, device=device, use_amp=use_amp,
                epoch=epoch, n_train_drugs=n_train_drugs,
            )
        else:
            train_loss, grad_norm = train_epoch_mse(
                model, optimizer, cell_features, cell_rows, fp_for_train,
                drug_idxs, targets, train_idx,
                bs=BATCH_SIZE, device=device, use_amp=use_amp,
            )
            pairs_per_batch_mean = float("nan")
            drugs_contributing_frac = float("nan")
            pair_label_pos_frac = float("nan")

        scheduler.step()

        # ----- val evaluation -----
        val_preds, val_tgts, val_didxs = eval_set(
            model, cell_features, cell_rows, fp_for_train,
            drug_idxs, targets, val_idx,
            bs=BATCH_SIZE * 2, device=device,
        )
        val_drug_names = np.array([idx_to_drug[i] for i in val_didxs])
        val_per_drug_r = float(mean_per_drug_r(val_preds, val_tgts, val_drug_names))
        val_loss = float(np.mean((val_preds - val_tgts) ** 2))

        # ----- train per-drug r (separate eval pass) -----
        tr_preds, tr_tgts, tr_didxs = eval_set(
            model, cell_features, cell_rows, fp_for_train,
            drug_idxs, targets, train_idx,
            bs=BATCH_SIZE * 2, device=device,
        )
        tr_drug_names = np.array([idx_to_drug[i] for i in tr_didxs])
        train_per_drug_r = float(mean_per_drug_r(tr_preds, tr_tgts, tr_drug_names))

        # ----- checkpoint selection -----
        if val_per_drug_r > best_val_r:
            best_val_r = val_per_drug_r
            best_val_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_epoch = epoch

        # ----- GPU memory -----
        if device == "cuda":
            gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_mem_gb = float("nan")

        elapsed = time.perf_counter() - t0
        lr_now = scheduler.get_last_lr()[0]

        # ----- sanity check after epoch 10 -----
        if epoch == 10 and train_per_drug_r <= 0.0:
            logger.warning(
                "[fold%d/%s] epoch 10 sanity: train_per_drug_r=%.4f <= 0 "
                "(possible dead gradients / broken loss / data pipeline error)",
                fold, condition, train_per_drug_r,
            )

        # ----- RankNet coverage check -----
        if is_ranknet and not np.isnan(drugs_contributing_frac):
            if drugs_contributing_frac < 0.90:
                logger.warning(
                    "[fold%d/%s] epoch %d: drugs_contributing_pairs_fraction=%.3f < 0.90",
                    fold, condition, epoch, drugs_contributing_frac,
                )

        row: Dict[str, Any] = {
            "fold": fold,
            "condition": condition,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_per_drug_r": train_per_drug_r,
            "val_per_drug_r": val_per_drug_r,
            "learning_rate": lr_now,
            "grad_norm_pre_clip": grad_norm,
            "epoch_time_s": elapsed,
            "gpu_memory_peak_gb": gpu_mem_gb,
            "pairs_per_batch_mean": pairs_per_batch_mean,
            "drugs_contributing_pairs_fraction": drugs_contributing_frac,
            "pair_label_positive_fraction": pair_label_pos_frac,
        }
        epoch_rows.append(row)

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "[fold%d/%s] ep %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                "train_r=%.4f  val_r=%.4f  lr=%.2e  %.1fs",
                fold, condition, epoch, N_EPOCHS,
                train_loss, val_loss, train_per_drug_r, val_per_drug_r,
                lr_now, elapsed,
            )

    # ----- save epoch metrics parquet -----
    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_parquet(
        logs_dir / f"fold{fold}_{condition}_epoch_metrics.parquet", index=False
    )

    # ----- save best checkpoint -----
    assert best_val_state is not None
    ckpt_path = ckpt_dir / f"fold{fold}_{condition}_best.pt"
    torch.save(
        {"state_dict": best_val_state, "epoch": best_val_epoch, "val_per_drug_r": best_val_r},
        ckpt_path,
    )

    # ----- test evaluation with best checkpoint -----
    model.load_state_dict(best_val_state)
    test_preds, test_tgts, test_didxs = eval_set(
        model, cell_features, cell_rows, fp_for_train,
        drug_idxs, targets, test_idx,
        bs=BATCH_SIZE * 2, device=device,
    )
    test_drug_names = np.array([idx_to_drug[i] for i in test_didxs])
    test_per_drug_r = float(mean_per_drug_r(test_preds, test_tgts, test_drug_names))

    # ----- save test predictions parquet -----
    test_cell_names = np.array([idx_to_cell[i] for i in test_idx])
    pred_df = pd.DataFrame({
        "drug_id": test_drug_names,
        "cell_id": test_cell_names,
        "y_true": test_tgts,
        "y_pred": test_preds,
    })
    pred_df.to_parquet(
        logs_dir / f"fold{fold}_{condition}_test_predictions.parquet", index=False
    )

    logger.info(
        "[fold%d/%s] DONE  best_epoch=%d  val_r=%.4f  test_per_drug_r=%.4f",
        fold, condition, best_val_epoch, best_val_r, test_per_drug_r,
    )

    return {
        "fold": fold,
        "condition": condition,
        "best_epoch": best_val_epoch,
        "val_per_drug_r_at_best": best_val_r,
        "test_per_drug_r": test_per_drug_r,
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

    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    # Logging to file
    (EXP_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log_tag = f"fold{run_single_fold}" if run_single_fold is not None else "all"
    fh = logging.FileHandler(EXP_DIR / "logs" / f"run_{log_tag}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logger.info(
        "06_objective_axis | started at %s | device=%s | fold=%s",
        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        device,
        run_single_fold if run_single_fold is not None else "all",
    )

    # ----- Load data -----
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    cl_idx = pd.read_parquet(DATA_DIR / "cell_line_index.parquet")

    name_to_depmap: Dict[str, str] = {}
    for depmap_id, row in cl_idx.iterrows():
        name_to_depmap[str(row["stripped_name"]).upper()] = str(depmap_id)

    logger.info("RNA: %s  Mutations: %s", rna.shape, mutations.shape)

    # ----- Load PASO 10-fold splits -----
    folds_raw = load_paso_folds(n_folds=K_FOLDS, paso_dir=PASO_FOLDS_DIR)

    # Build unified pair dataframe and drug/cell indices
    full_df, key_to_idx = build_pair_index(
        folds_raw, name_to_depmap, rna.index, mutations.index
    )
    logger.info(
        "Full dataset: %d pairs, %d cells, %d drugs",
        len(full_df),
        full_df["depmap_id"].nunique(),
        full_df["drug_name"].nunique(),
    )

    all_drugs = sorted(full_df["drug_name"].unique())
    drug_to_idx: Dict[str, int] = {d: i for i, d in enumerate(all_drugs)}
    idx_to_drug: Dict[int, str] = {i: d for d, i in drug_to_idx.items()}

    # Build raw cell feature arrays (will be PCA-compressed per fold)
    all_cells = sorted(full_df["depmap_id"].unique())
    cell_to_row: Dict[str, int] = {c: i for i, c in enumerate(all_cells)}

    rna_raw = rna.loc[all_cells].values.astype(np.float32)
    mut_raw = mutations.loc[all_cells].values.astype(np.float32)

    cell_rows = np.array([cell_to_row[c] for c in full_df["depmap_id"]], dtype=np.int32)
    drug_idxs_arr = np.array([drug_to_idx[d] for d in full_df["drug_name"]], dtype=np.int32)
    targets = full_df["ic50"].values.astype(np.float32)

    # Map dataset row index → depmap_id (for predictions parquet)
    row_to_depmap: Dict[int, str] = {i: row["depmap_id"] for i, row in full_df.iterrows()}

    # ----- Drug fingerprints (Morgan 2048-bit) -----
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
    logger.info("Morgan FP matrix: %s", fp_matrix.shape)

    # ----- Cross-validation -----
    fold_results: List[Dict[str, Any]] = []
    folds_to_run = [run_single_fold] if run_single_fold is not None else list(range(K_FOLDS))
    logger.info("Running folds: %s", folds_to_run)

    for fold_i in folds_to_run:
        t_fold_start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fold_i + 1, K_FOLDS)
        logger.info("=" * 60)

        train_df_raw, test_df_raw = folds_raw[fold_i]
        full_train_idx = map_fold_indices(train_df_raw, key_to_idx, name_to_depmap)
        test_idx = map_fold_indices(test_df_raw, key_to_idx, name_to_depmap)
        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i)

        logger.info("  train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

        # PCA compression — fit on train cells only (no leakage)
        train_cell_rows_unique = cell_rows[train_idx]
        rna_c, mut_c = compress_cell(rna_raw, mut_raw, train_cell_rows_unique)
        cell_features = np.concatenate([rna_c, mut_c], axis=1)  # (all_cells, 750)
        logger.info("  Cell features (post-PCA): %s", cell_features.shape)

        # Build idx_to_cell mapping for this fold's test set
        idx_to_cell: Dict[int, str] = row_to_depmap

        for condition in CONDITIONS:
            logger.info("  --- Condition: %s ---", condition)
            t_cond_start = time.perf_counter()
            fold_cond_result = train_condition(
                condition=condition,
                cell_features=cell_features,
                cell_rows=cell_rows,
                fp_matrix=fp_matrix,
                drug_idxs=drug_idxs_arr,
                targets=targets,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                idx_to_drug=idx_to_drug,
                idx_to_cell=idx_to_cell,
                fold=fold_i,
                device=device,
            )
            fold_cond_result["total_train_time_h"] = (
                time.perf_counter() - t_cond_start
            ) / 3600.0
            fold_results.append(fold_cond_result)

            # Clear VRAM between conditions
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        fold_elapsed = time.perf_counter() - t_fold_start
        logger.info("FOLD %d done in %.1f min", fold_i, fold_elapsed / 60.0)

        # If sharding: write fold results immediately and stop
        if run_single_fold is not None:
            report_dir = EXP_DIR / "report" / "data"
            report_dir.mkdir(parents=True, exist_ok=True)
            shard_path = report_dir / f"fold_{fold_i:02d}_results.json"
            with shard_path.open("w") as f:
                json.dump(fold_results, f, indent=2)
            logger.info("Shard written to %s", shard_path)
            return

    # ----- Aggregate results -----
    results_by_condition: Dict[str, Dict[str, Any]] = {}

    for cond in CONDITIONS:
        fold_rs = [
            r["test_per_drug_r"]
            for r in fold_results
            if r["condition"] == cond
        ]
        results_by_condition[cond] = {
            "mean": float(np.mean(fold_rs)),
            "std": float(np.std(fold_rs)),
            "folds": [float(r) for r in fold_rs],
        }

    # Compute Δ vs no_drug baseline
    base_mean = results_by_condition["mlp_mse_no_drug"]["mean"]
    for cond in ("mlp_mse_morgan", "mlp_ranknet_morgan"):
        results_by_condition[cond]["delta"] = float(
            results_by_condition[cond]["mean"] - base_mean
        )

    # RankNet vs MSE-morgan delta
    ranknet_vs_mse_delta = float(
        results_by_condition["mlp_ranknet_morgan"]["mean"]
        - results_by_condition["mlp_mse_morgan"]["mean"]
    )
    results_by_condition["ranknet_vs_mse_delta"] = ranknet_vs_mse_delta  # type: ignore[assignment]

    # ----- Validation checks -----
    mse_delta = results_by_condition["mlp_mse_morgan"]["delta"]
    if abs(mse_delta - 0.003) > 0.005:
        logger.warning(
            "Validation check FAILED: mlp_mse_morgan Δ=%.4f deviates from expected "
            "Ridge Δ≈0.003 by more than ±0.005 (|%.4f - 0.003| = %.4f). "
            "Investigate before concluding on RankNet.",
            mse_delta, mse_delta, abs(mse_delta - 0.003),
        )

    # ----- Write output -----
    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(results_by_condition, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)

    # ----- Summary log -----
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for cond in CONDITIONS:
        r = results_by_condition[cond]
        delta_str = f"  Δ={r.get('delta', 0.0):+.4f}" if "delta" in r else ""
        logger.info(
            "  %-22s  mean=%.4f ± %.4f%s  folds=%s",
            cond,
            r["mean"],
            r["std"],
            delta_str,
            " ".join(f"{v:.3f}" for v in r["folds"]),
        )
    logger.info(
        "  ranknet_vs_mse_delta:  %+.4f", ranknet_vs_mse_delta
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
