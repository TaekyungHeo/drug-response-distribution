"""Phase 12: Metric diagnostic — global vs per-drug Pearson r.

Question: Is the drug-blind ceiling at 0.52 a MODEL limit or a METRIC artifact?

If the model predicts "drug X tends to have high IC50" from fingerprint alone
(drug identity), that inflates global Pearson r even if within-drug cell ranking
is poor. Per-drug Pearson r removes the drug identity signal and measures only
within-drug discrimination: "for this specific drug, which cells are sensitive?"

Diagnostic:
  - Global r: Pearson r across all test pairs (what we've been reporting)
  - Per-drug r: Pearson r within each test drug, then averaged
  - Gap: global r − per-drug r = contribution of drug identity to the metric

Also tests: per-drug z-scored training (normalize IC50 per drug before training).
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr as scipy_pearsonr

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

from src.data.drug_features import get_drug_fingerprints
from src.evaluation.metrics import evaluate
from src.evaluation.per_drug import per_drug_r as _canonical_per_drug_r
from src.models.transformer_encoder import TransformerEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

N_EPOCHS = 50
WARMUP_EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 512
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
MODALITY_DROPOUT_P = 0.3
OMICS = ["rna", "mutations"]
K_FOLDS = 5

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"


def compute_per_drug_r(predictions: np.ndarray, targets: np.ndarray, drug_names: np.ndarray) -> Dict[str, Any]:
    """Compute global r and per-drug r."""
    global_r = float(scipy_pearsonr(predictions, targets)[0])
    rs_dict = _canonical_per_drug_r(predictions, targets, drug_names, min_cells=5)
    per_drug_rs = list(rs_dict.values())
    per_drug_details = [{"drug": d, "r": r, "n": int((drug_names == d).sum())} for d, r in rs_dict.items()]
    return {
        "global_r": global_r,
        "per_drug_r_mean": float(np.mean(per_drug_rs)),
        "per_drug_r_median": float(np.median(per_drug_rs)),
        "per_drug_r_std": float(np.std(per_drug_rs)),
        "n_drugs_evaluated": len(per_drug_rs),
        "gap": global_r - float(np.mean(per_drug_rs)),
        "per_drug_details": sorted(per_drug_details, key=lambda x: x["r"]),
    }


# Reuse Phase 11's data loading infrastructure
import queue, threading


class _Prefetcher:
    def __init__(self, concat_np, cell_rows, drug_idxs, fp_matrix, targets, indices, bs, device):
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


def train_and_predict(
    concat_np, cell_rows, drug_idxs, fp_matrix, targets,
    feature_dims, train_idx, val_idx, test_idx,
    n_epochs, bs, lr, device, fold_name, use_zscore=False,
    drug_names_arr=None,
):
    """Train one fold, return test predictions + targets + drug names."""
    train_targets = targets.copy()
    if use_zscore and drug_names_arr is not None:
        # Z-score targets per drug (training drugs only)
        train_drug_names = drug_names_arr[train_idx]
        for drug in np.unique(train_drug_names):
            mask = train_drug_names == drug
            global_mask = np.isin(np.arange(len(targets)), train_idx[mask])
            vals = targets[global_mask]
            if vals.std() > 1e-8:
                mean, std = vals.mean(), vals.std()
                train_targets[global_mask] = (vals - mean) / std
        # Also z-score val targets using same per-drug stats from train
        # (for fair validation metric)

    model = TransformerEncoder(
        feature_dims=feature_dims, modality_order=OMICS,
        drug_fp_dim=fp_matrix.shape[1],
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, modality_dropout_p=MODALITY_DROPOUT_P,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - WARMUP_EPOCHS), eta_min=lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    criterion = nn.MSELoss()

    steps = len(train_idx) // bs
    best_val_r, best_state = -np.inf, None

    for epoch in range(1, n_epochs + 1):
        model.train()
        pf = _Prefetcher(concat_np, cell_rows, drug_idxs, fp_matrix, train_targets, train_idx, bs, device)
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

        # Val
        model.eval()
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(val_idx), bs * 2):
                chunk = val_idx[i:i+bs*2]
                rows = cell_rows[chunk]
                x = torch.from_numpy(concat_np[rows].copy()).to(device)
                fp_batch = torch.from_numpy(fp_matrix[drug_idxs[chunk]].copy()).to(device)
                pred = model(x, fp_batch)
                preds_list.append(pred.cpu().numpy())
        val_preds = np.concatenate(preds_list)
        val_r = float(scipy_pearsonr(train_targets[val_idx], val_preds)[0])

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            logger.info("[%s] ep %3d/%d  val_r=%.4f", fold_name, epoch, n_epochs, val_r)

    # Test with best model — predict on ORIGINAL targets (not z-scored)
    model.load_state_dict(best_state)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(test_idx), bs * 2):
            chunk = test_idx[i:i+bs*2]
            rows = cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(device)
            fp_batch = torch.from_numpy(fp_matrix[drug_idxs[chunk]].copy()).to(device)
            pred = model(x, fp_batch)
            preds_list.append(pred.cpu().numpy())
    test_preds = np.concatenate(preds_list)
    test_targets_orig = targets[test_idx]  # Always evaluate on original IC50
    test_drugs = drug_names_arr[test_idx]

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return test_preds, test_targets_orig, test_drugs


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["standard", "per_drug_zscore", "both"], default="both")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/mps/cuda)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXP_DIR / "results" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "logs").mkdir(exist_ok=True)

    fh = logging.FileHandler(EXP_DIR / "logs" / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("Phase 12 Metric Diagnostic | run_dir=%s", run_dir)

    # Load data
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    cl_idx = pd.read_parquet(DATA_DIR / "cell_line_index.parquet")

    name_to_depmap: Dict[str, str] = {}
    for depmap_id, row in cl_idx.iterrows():
        name_to_depmap[row["stripped_name"].upper()] = str(depmap_id)

    feature_dims = {"rna": rna.shape[1], "mutations": mutations.shape[1]}

    # Load PASO fold data
    all_folds_raw = []
    for fold_i in range(K_FOLDS):
        train_df = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{fold_i}.csv")
        test_df = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{fold_i}.csv")
        all_folds_raw.append((train_df, test_df))

    all_pairs = pd.concat([pd.concat([tr, te]) for tr, te in all_folds_raw]).drop_duplicates()
    available_cells = set(rna.index) & set(mutations.index)

    # Build full dataset
    all_drugs_set = set()
    valid_rows = []
    for _, row in all_pairs.iterrows():
        ccl = str(row["cell_line"]).upper()
        depmap = name_to_depmap.get(ccl)
        drug = row["drug"]
        if depmap and depmap in available_cells:
            valid_rows.append({"depmap_id": depmap, "drug_name": drug, "ic50": float(row["IC50"])})
            all_drugs_set.add(drug)

    full_df = pd.DataFrame(valid_rows)
    all_drugs = sorted(all_drugs_set)
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)

    all_cells = sorted(full_df["depmap_id"].unique())
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    concat_np = np.concatenate([rna_arr, mut_arr], axis=1)

    cell_rows = np.array([cell_to_row[r] for r in full_df["depmap_id"]], dtype=np.int32)
    drug_idxs_arr = np.array([drug_to_idx[d] for d in full_df["drug_name"]], dtype=np.int32)
    targets = full_df["ic50"].values.astype(np.float32)
    drug_names_arr = np.array(full_df["drug_name"].tolist())

    pair_keys = list(zip(full_df["depmap_id"], full_df["drug_name"]))
    key_to_idx = {k: i for i, k in enumerate(pair_keys)}

    logger.info("Dataset: %d pairs, %d cells, %d drugs", len(full_df), len(all_cells), len(all_drugs))

    # Two conditions: standard training vs per-drug z-scored training
    all_conditions = [
        ("standard", False),
        ("per_drug_zscore", True),
    ]
    if args.condition == "both":
        conditions = all_conditions
    else:
        conditions = [c for c in all_conditions if c[0] == args.condition]

    results: Dict[str, Any] = {}
    results_path = run_dir / "results.json"

    for cond_name, use_zscore in conditions:
        logger.info("=" * 60)
        logger.info("Condition: %s", cond_name)
        cond_results = {"folds": [], "global_rs": [], "per_drug_rs": []}

        for fold_i in range(K_FOLDS):
            train_df, test_df = all_folds_raw[fold_i]

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

            rng = np.random.default_rng(42 + fold_i)
            perm = rng.permutation(len(full_train_idx))
            n_val = len(full_train_idx) // 10
            val_idx = full_train_idx[perm[:n_val]]
            train_idx = full_train_idx[perm[n_val:]]

            logger.info("=== %s Fold %d/%d | train=%d val=%d test=%d ===",
                        cond_name, fold_i + 1, K_FOLDS, len(train_idx), len(val_idx), len(test_idx))

            test_preds, test_targets, test_drugs = train_and_predict(
                concat_np, cell_rows, drug_idxs_arr, fp_matrix, targets,
                feature_dims, train_idx, val_idx, test_idx,
                N_EPOCHS, BATCH_SIZE, LR, device,
                f"{cond_name}_fold{fold_i}",
                use_zscore=use_zscore,
                drug_names_arr=drug_names_arr,
            )

            metrics = compute_per_drug_r(test_preds, test_targets, test_drugs)
            logger.info("  %s fold%d | global_r=%.4f  per_drug_r=%.4f  gap=%.4f  (%d drugs)",
                        cond_name, fold_i, metrics["global_r"], metrics["per_drug_r_mean"],
                        metrics["gap"], metrics["n_drugs_evaluated"])

            cond_results["folds"].append(metrics)
            cond_results["global_rs"].append(metrics["global_r"])
            cond_results["per_drug_rs"].append(metrics["per_drug_r_mean"])

            with results_path.open("w") as f:
                results[cond_name] = cond_results
                json.dump(results, f, indent=2, default=str)

        cond_results["global_r_mean"] = float(np.mean(cond_results["global_rs"]))
        cond_results["per_drug_r_mean_mean"] = float(np.mean(cond_results["per_drug_rs"]))
        cond_results["gap_mean"] = cond_results["global_r_mean"] - cond_results["per_drug_r_mean_mean"]
        results[cond_name] = cond_results

        with results_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("--- %s summary ---", cond_name)
        logger.info("  Global r mean: %.4f", cond_results["global_r_mean"])
        logger.info("  Per-drug r mean: %.4f", cond_results["per_drug_r_mean_mean"])
        logger.info("  Gap (drug identity contribution): %.4f", cond_results["gap_mean"])

    # Final comparison
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    for cond_name in ["standard", "per_drug_zscore"]:
        c = results[cond_name]
        logger.info("  %s: global=%.4f  per_drug=%.4f  gap=%.4f",
                    cond_name, c["global_r_mean"], c["per_drug_r_mean_mean"], c["gap_mean"])


if __name__ == "__main__":
    main()