"""05_dataset_robustness/run_transformer.py: PRISM TransformerEncoder ablation.

morgan_fp vs no_drug, 5-fold drug-blind CV, 200 epochs.
Parallels 03_model_robustness Part A but on the PRISM dataset.

Data: data/processed/prism_drug_response.parquet (ln_ic50)
Cell features: raw RNA + mutations (no PCA — TransformerEncoder handles dimensionality).

Telemetry per epoch:
  - logs/fold{k}_{condition}_epoch_metrics.parquet
  - logs/fold{k}_{condition}_test_predictions.parquet
  - checkpoints/fold{k}_{condition}_best.pt

Output: report/data/transformer_metrics.json
"""

from __future__ import annotations

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

from src.data.drug_features import get_drug_fingerprints
from src.data.prism import load_prism, preprocess_prism
from src.evaluation.per_drug import mean_per_drug_r
from src.models.transformer_encoder import TransformerEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "processed"
EXP_DIR = Path(__file__).parents[1]

# Training config (matches 03_model_robustness Part A)
N_EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 256
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
MODALITY_DROPOUT_P = 0.3
WARMUP_EPOCHS = 10
VAL_DRUG_FRAC = 0.10

K_FOLDS = 5
MIN_CELLS_PER_DRUG = 50
MIN_CELLS_EVAL = 5
FOLD_SEED = 42
MODEL_SEED = 0


def make_drug_folds(drug_names: List[str], n_folds: int = 5, seed: int = FOLD_SEED) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(drug_names))
    parts = np.array_split(shuffled, n_folds)
    return [[drug_names[i] for i in part] for part in parts]


# ---------------------------------------------------------------------------
# Prefetch batch loader
# ---------------------------------------------------------------------------

class _Prefetcher:
    """Background-thread batch prefetcher for (omics, drug_fp, target) batches."""

    def __init__(
        self,
        concat_np: np.ndarray,
        cell_rows: np.ndarray,
        drug_idxs: np.ndarray,
        fp_matrix: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        device: str,
    ) -> None:
        self._c = concat_np
        self._cr = cell_rows
        self._di = drug_idxs
        self._fp = fp_matrix
        self._t = targets
        self._idx = indices
        self._bs = batch_size
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
                batch_idx = self._idx[perm[i:i + self._bs]]
                i += self._bs
                rows = self._cr[batch_idx]
                x = torch.from_numpy(self._c[rows].copy()).to(self._dev)
                fp = torch.from_numpy(self._fp[self._di[batch_idx]].copy()).to(self._dev)
                y = torch.from_numpy(self._t[batch_idx].copy()).to(self._dev)
                self._q.put((x, fp, y), timeout=60)
        except Exception as exc:
            self._err = exc

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._err:
            raise RuntimeError(str(self._err))
        return self._q.get(timeout=60)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_set(
    model: TransformerEncoder,
    concat_np: np.ndarray,
    cell_rows: np.ndarray,
    drug_idxs: np.ndarray,
    fp_matrix: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on given indices. Returns (preds, targets, drug_idxs)."""
    model.eval()
    preds_list, tgts_list, didx_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size * 2):
            chunk = indices[i:i + batch_size * 2]
            rows = cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(device)
            fp = torch.from_numpy(fp_matrix[drug_idxs[chunk]].copy()).to(device)
            pred = model(x, fp)
            preds_list.append(pred.cpu().numpy())
            tgts_list.append(targets[chunk])
            didx_list.append(drug_idxs[chunk])
    return np.concatenate(preds_list), np.concatenate(tgts_list), np.concatenate(didx_list)


# ---------------------------------------------------------------------------
# Fold trainer
# ---------------------------------------------------------------------------

def train_condition(
    k: int,
    condition: str,
    fold_drugs: List[List[str]],
    dr: pd.DataFrame,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    fp_matrix: np.ndarray,
    drug_to_idx: Dict[str, int],
    idx_to_drug: Dict[int, str],
    device: str,
) -> float:
    """Train one (fold, condition) pair. Returns test per-drug r.

    Saves:
      - logs/fold{k}_{condition}_epoch_metrics.parquet
      - logs/fold{k}_{condition}_test_predictions.parquet
      - checkpoints/fold{k}_{condition}_best.pt
    """
    available_cells = set(rna.index) & set(mutations.index)

    test_drug_set = set(fold_drugs[k])
    train_drug_set_full = set(d for i, drugs in enumerate(fold_drugs) if i != k for d in drugs)

    # Carve out drug-blind val from training drugs
    train_drugs_sorted = sorted(train_drug_set_full)
    rng = np.random.default_rng(42 + k)
    n_val_drugs = max(1, int(len(train_drugs_sorted) * VAL_DRUG_FRAC))
    shuffled_train = rng.permutation(len(train_drugs_sorted))
    val_drug_set = {train_drugs_sorted[i] for i in shuffled_train[:n_val_drugs]}
    train_drug_set = {train_drugs_sorted[i] for i in shuffled_train[n_val_drugs:]}

    # Build DataFrames
    train_df = dr[dr["drug_name"].isin(train_drug_set) & dr["depmap_id"].isin(available_cells)].copy()
    val_df = dr[dr["drug_name"].isin(val_drug_set) & dr["depmap_id"].isin(available_cells)].copy()
    test_df_raw = dr[dr["drug_name"].isin(test_drug_set) & dr["depmap_id"].isin(available_cells)].copy()

    # Filter test drugs with < MIN_CELLS_PER_DRUG
    test_counts = test_df_raw.groupby("drug_name").size()
    valid_test_drugs = test_counts[test_counts >= MIN_CELLS_PER_DRUG].index
    test_df = test_df_raw[test_df_raw["drug_name"].isin(valid_test_drugs)].copy()

    logger.info(
        "  [fold%d %s] train_drugs=%d val_drugs=%d test_drugs=%d  "
        "pairs: train=%d val=%d test=%d",
        k, condition,
        len(train_drug_set), len(val_drug_set), len(valid_test_drugs),
        len(train_df), len(val_df), len(test_df),
    )

    # Build unified cell array (raw, no PCA)
    all_cells = sorted(set(train_df["depmap_id"]) | set(val_df["depmap_id"]) | set(test_df["depmap_id"]))
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    concat_np = np.concatenate([rna_arr, mut_arr], axis=1)  # (n_cells, n_genes + n_mut)

    def make_arrays(df: pd.DataFrame):
        cell_idx = np.array([cell_to_row[c] for c in df["depmap_id"]], dtype=np.int32)
        drug_idx = np.array([drug_to_idx[d] for d in df["drug_name"]], dtype=np.int32)
        targets = df["response"].values.astype(np.float32)
        return cell_idx, drug_idx, targets

    train_cell_idx, train_drug_idx, train_targets = make_arrays(train_df)
    val_cell_idx, val_drug_idx, val_targets = make_arrays(val_df)
    test_cell_idx, test_drug_idx, test_targets = make_arrays(test_df)

    # Remap indices to be relative (cell_rows and drug_idxs arrays are built per-fold)
    cell_rows = np.concatenate([train_cell_idx, val_cell_idx, test_cell_idx])
    drug_idxs_full = np.concatenate([train_drug_idx, val_drug_idx, test_drug_idx])
    targets_full = np.concatenate([train_targets, val_targets, test_targets])

    n_train = len(train_df)
    n_val = len(val_df)
    train_idx_global = np.arange(n_train, dtype=np.int64)
    val_idx_global = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx_global = np.arange(n_train + n_val, n_train + n_val + len(test_df), dtype=np.int64)

    # Fingerprint matrix: full if morgan_fp, zero if no_drug
    n_total_drugs = len(drug_to_idx)
    if condition == "morgan_fp":
        fp_used = fp_matrix  # (n_total_drugs, 2048)
    else:  # no_drug
        fp_used = np.zeros((n_total_drugs, fp_matrix.shape[1]), dtype=np.float32)

    # Model
    torch.manual_seed(MODEL_SEED)
    n_rna = rna_arr.shape[1]
    n_mut = mut_arr.shape[1]
    feature_dims = {"rna": n_rna, "mutations": n_mut}
    model = TransformerEncoder(
        feature_dims=feature_dims,
        modality_order=["rna", "mutations"],
        drug_fp_dim=fp_matrix.shape[1],
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        modality_dropout_p=MODALITY_DROPOUT_P,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Model params: %d", n_params)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, N_EPOCHS - WARMUP_EPOCHS), eta_min=LR * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )
    criterion = nn.MSELoss()

    steps_per_epoch = max(1, len(train_idx_global) // BATCH_SIZE)

    best_val_r = -np.inf
    best_state: Optional[Dict] = None
    best_epoch = 0

    epoch_records: List[Dict[str, Any]] = []

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()

        pf = _Prefetcher(
            concat_np, cell_rows, drug_idxs_full, fp_used, targets_full,
            train_idx_global, BATCH_SIZE, device,
        )
        train_losses = []
        for _ in range(steps_per_epoch):
            x, fp, y = next(pf)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, fp)
            loss = criterion(pred, y)
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            )
            optimizer.step()
            train_losses.append(loss.item())
        pf.stop()
        scheduler.step()

        train_loss = float(np.mean(train_losses))
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.perf_counter() - t0

        # Val evaluation
        val_preds, val_tgts, val_didxs = eval_set(
            model, concat_np, cell_rows, drug_idxs_full, fp_used, targets_full,
            val_idx_global, BATCH_SIZE, device,
        )
        val_loss = float(criterion(torch.tensor(val_preds), torch.tensor(val_tgts)).item())
        val_drug_names = np.array([idx_to_drug[i] for i in val_didxs])
        val_per_drug_r = mean_per_drug_r(val_preds, val_tgts, val_drug_names, min_cells=MIN_CELLS_EVAL)

        # Train per-drug r (sample to save time: subsample up to 10k train pairs)
        tr_sample_size = min(len(train_idx_global), 10000)
        rng_ep = np.random.default_rng(epoch)
        tr_sample = rng_ep.choice(train_idx_global, size=tr_sample_size, replace=False)
        tr_preds, tr_tgts, tr_didxs = eval_set(
            model, concat_np, cell_rows, drug_idxs_full, fp_used, targets_full,
            tr_sample, BATCH_SIZE, device,
        )
        tr_drug_names = np.array([idx_to_drug[i] for i in tr_didxs])
        train_per_drug_r = mean_per_drug_r(tr_preds, tr_tgts, tr_drug_names, min_cells=MIN_CELLS_EVAL)

        # GPU memory
        gpu_mem_peak_gb = 0.0
        gpu_mem_reserved_gb = 0.0
        if device == "cuda":
            gpu_mem_peak_gb = torch.cuda.max_memory_allocated() / 1e9
            gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
            torch.cuda.reset_peak_memory_stats()
        elif device == "mps":
            gpu_mem_peak_gb = torch.mps.current_allocated_memory() / 1e9

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_per_drug_r": float(train_per_drug_r),
            "val_per_drug_r": float(val_per_drug_r),
            "learning_rate": current_lr,
            "grad_norm_pre_clip": grad_norm,
            "epoch_time_s": epoch_time,
            "gpu_memory_peak_gb": gpu_mem_peak_gb,
            "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
        }
        epoch_records.append(epoch_record)

        if val_per_drug_r > best_val_r:
            best_val_r = val_per_drug_r
            best_epoch = epoch
            best_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "  [fold%d %s] ep %3d/%d  train_loss=%.4f  val_r=%.4f  best_ep=%d  %.1fs",
                k, condition, epoch, N_EPOCHS, train_loss, val_per_drug_r, best_epoch, epoch_time,
            )

    # Save epoch metrics parquet
    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    epoch_df = pd.DataFrame(epoch_records)
    epoch_df.to_parquet(logs_dir / f"fold{k}_{condition}_epoch_metrics.parquet", index=False)

    # Load best checkpoint and evaluate on test
    assert best_state is not None
    model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_dir = EXP_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / f"fold{k}_{condition}_best.pt")

    test_preds, test_tgts, test_didxs = eval_set(
        model, concat_np, cell_rows, drug_idxs_full, fp_used, targets_full,
        test_idx_global, BATCH_SIZE, device,
    )
    test_drug_names = np.array([idx_to_drug[i] for i in test_didxs])
    test_cell_names = np.array([all_cells[cell_rows[i]] for i in test_idx_global])

    test_r = mean_per_drug_r(test_preds, test_tgts, test_drug_names, min_cells=MIN_CELLS_EVAL)

    logger.info(
        "  [fold%d %s] DONE  best_epoch=%d  val_r_at_best=%.4f  test_per_drug_r=%.4f",
        k, condition, best_epoch, best_val_r, test_r,
    )

    # Save test predictions parquet
    pred_df = pd.DataFrame({
        "depmap_id": test_cell_names,
        "drug_name": test_drug_names,
        "y_true": test_tgts,
        "y_pred": test_preds,
    })
    pred_df.to_parquet(logs_dir / f"fold{k}_{condition}_test_predictions.parquet", index=False)

    return float(test_r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("05_dataset_robustness/run_transformer: PRISM TransformerEncoder ablation")

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    # Load PRISM
    df_raw = load_prism(DATA_DIR)

    # Load cell features (raw, no PCA for Transformer)
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  Mutations: %s", rna.shape, mutations.shape)

    rna_cell_ids = set(rna.index) & set(mutations.index)
    dr = pd.DataFrame(preprocess_prism(df_raw, rna_cell_ids)[0])

    # Intersect rna/mutations to the PRISM cells to save memory
    prism_cells = set(dr["depmap_id"].unique())
    # Use intersection of all three
    usable_cells = sorted(prism_cells & set(rna.index) & set(mutations.index))
    rna = rna.loc[usable_cells]
    mutations = mutations.loc[usable_cells]
    logger.info("Usable cells (PRISM ∩ RNA ∩ mutations): %d", len(usable_cells))

    dr = dr[dr["depmap_id"].isin(usable_cells)].reset_index(drop=True)

    # Build drug index
    drug_names_all = sorted(dr["drug_name"].unique())  # type: ignore[union-attr]
    drug_to_idx = {d: i for i, d in enumerate(drug_names_all)}
    idx_to_drug = {i: d for d, i in drug_to_idx.items()}
    logger.info("Final: %d drugs, %d cells", len(drug_names_all), len(usable_cells))

    # Morgan fingerprints
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
    logger.info("Fingerprint matrix: %s", fp_matrix.shape)

    # Drug folds
    fold_drugs = make_drug_folds(drug_names_all, n_folds=K_FOLDS, seed=FOLD_SEED)
    logger.info("Fold sizes: %s", [len(f) for f in fold_drugs])

    conditions = ["morgan_fp", "no_drug"]
    results: Dict[str, List[float]] = {c: [] for c in conditions}

    for k in range(K_FOLDS):
        for condition in conditions:
            logger.info("=== Fold %d/%d | %s ===", k, K_FOLDS - 1, condition)
            test_r = train_condition(
                k=k,
                condition=condition,
                fold_drugs=fold_drugs,
                dr=dr,  # type: ignore[arg-type]
                rna=rna,
                mutations=mutations,
                fp_matrix=fp_matrix,
                drug_to_idx=drug_to_idx,
                idx_to_drug=idx_to_drug,
                device=device,
            )
            results[condition].append(test_r)

            # Clear GPU cache between runs
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    # Summary
    logger.info("=" * 60)
    output: Dict[str, Any] = {}
    for condition in conditions:
        folds = results[condition]
        mean_r = float(np.mean(folds))
        std_r = float(np.std(folds))
        logger.info("%s: mean=%.4f ± %.4f  folds=%s",
                    condition, mean_r, std_r, [round(r, 4) for r in folds])
        output[condition] = {
            "mean": mean_r,
            "std": std_r,
            "folds": [float(r) for r in folds],
        }

    delta = output["morgan_fp"]["mean"] - output["no_drug"]["mean"]
    output["delta_morgan_vs_no_drug"] = delta
    logger.info("Delta (morgan_fp - no_drug): %.4f", delta)

    # Write output
    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "transformer_metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
