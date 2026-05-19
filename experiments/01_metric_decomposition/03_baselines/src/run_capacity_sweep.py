"""Stage 2A: Fixed-capacity MLP sweep (RNA-only) — capacity does not confound modality.

Architectures: Small [128→64→1] / Medium [512→256→64→1] / Large [2048→512→128→1]
Hyperparameter grid: dropout [0.1, 0.3, 0.5] × weight_decay [0, 1e-4, 1e-3]
Val metric: per-drug r (primary metric)
Training: LR cosine [1e-3→1e-5], max_epochs=300, early stopping patience=30

Best config per capacity level: selected by val per-drug r on mixed-set split.
Each winning config evaluated on all three splits.

If Large ≈ Medium on per-drug r → capacity is not the bottleneck.

Best medium config saved to results/stage2/capacity_sweep_best_configs.json
for use by run_modality_ablation.py.

Usage:
    uv run python3 experiments/01_metric_decomposition/03_baselines/src/run_capacity_sweep.py
    uv run python3 experiments/01_metric_decomposition/03_baselines/src/run_capacity_sweep.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).parents[4]
EXPERIMENT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from src.data.omics_utils import build_pair_features, load_omics, z_score_normalize
from src.data.splits import cell_blind_split, drug_blind_split, mixed_set_split
from src.evaluation.metrics import evaluate_full
from src.train import predict_batched, train_mlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "experiments" / "01_metric_decomposition" / "03_baselines" / "results" / "stage2"
REPORT_DATA = REPO_ROOT / "experiments" / "01_metric_decomposition" / "03_baselines" / "report" / "data" / "metrics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096 if DEVICE == "cuda" else 2048
MAX_EPOCHS = 300
PATIENCE = 30

CAPACITY_GRID: dict[str, list[int]] = {
    "small":  [128, 64],
    "medium": [512, 256, 64],
    "large":  [2048, 512, 128],
}
DROPOUT_GRID = [0.1, 0.3, 0.5]
WD_GRID = [0.0, 1e-4, 1e-3]

SPLIT_FNS = {
    "mixed_set": mixed_set_split,
    "cell_blind": cell_blind_split,
    "drug_blind": drug_blind_split,
}


def sweep_capacity(
    dr: pd.DataFrame,
    pair_X: np.ndarray,
    smoke: bool = False,
) -> dict:
    logger.info("=== Stage 2A: capacity sweep (RNA-only, mixed-set val) ===")
    train_idx, val_idx, _ = mixed_set_split(dr, seed=42)
    Xtr0, Xvl0, _ = z_score_normalize(pair_X[train_idx], pair_X[val_idx], pair_X[val_idx])
    ytr0 = dr.iloc[train_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    yvl0 = dr.iloc[val_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    val_drugs0 = dr.iloc[val_idx]["drug_name"].to_numpy()

    dropouts = DROPOUT_GRID[:1] if smoke else DROPOUT_GRID
    wds = WD_GRID[:1] if smoke else WD_GRID
    max_ep = 2 if smoke else MAX_EPOCHS
    caps = {k: v for k, v in list(CAPACITY_GRID.items())[:1]} if smoke else CAPACITY_GRID

    best_configs: dict[str, dict] = {}
    for cap_name, hidden_dims in caps.items():
        best_vr, best_do, best_wd = -np.inf, dropouts[0], wds[0]
        for do in dropouts:
            for wd in wds:
                _, _, vr, _ = train_mlp(
                    Xtr0, ytr0, Xvl0, yvl0, val_drugs0,
                    hidden_dims, do, wd, DEVICE,
                    max_epochs=max_ep, patience=PATIENCE, batch_size=BATCH_SIZE,
                )
                if not np.isnan(vr) and vr > best_vr:
                    best_vr, best_do, best_wd = vr, do, wd
        best_configs[cap_name] = {"dropout": best_do, "weight_decay": best_wd, "val_per_drug_r": float(best_vr)}
        logger.info("  [%s] best dropout=%.1f wd=%g val_per_drug_r=%.4f", cap_name, best_do, best_wd, best_vr)

    # Evaluate best config per capacity on all splits
    eval_results: dict[str, dict] = {}
    for cap_name, hidden_dims in caps.items():
        cfg = best_configs[cap_name]
        eval_results[cap_name] = {"config": cfg, "splits": {}}
        for split_name, split_fn in SPLIT_FNS.items():
            tr_i, vl_i, te_i = split_fn(dr, seed=42)
            Xtr, Xvl, Xte = z_score_normalize(pair_X[tr_i], pair_X[vl_i], pair_X[te_i])
            ytr = dr.iloc[tr_i]["ln_ic50"].to_numpy(dtype=np.float32)
            yvl = dr.iloc[vl_i]["ln_ic50"].to_numpy(dtype=np.float32)
            val_drugs = dr.iloc[vl_i]["drug_name"].to_numpy()
            model, curve, _, best_epoch = train_mlp(
                Xtr, ytr, Xvl, yvl, val_drugs,
                hidden_dims, float(cfg["dropout"]), float(cfg["weight_decay"]), DEVICE,
                max_epochs=max_ep, patience=PATIENCE, batch_size=BATCH_SIZE,
            )
            preds = predict_batched(model, Xte, DEVICE, BATCH_SIZE)
            y_te = dr.iloc[te_i]["ln_ic50"].to_numpy(dtype=np.float32)
            m = dict(evaluate_full(
                y_te, preds,
                dr.iloc[te_i]["drug_name"].to_numpy(),
                dr.iloc[te_i]["depmap_id"].to_numpy(),
            ))
            m["val_per_drug_r_curve"] = [round(v, 5) if not np.isnan(v) else None for v in curve]
            m["best_epoch"] = best_epoch
            m["converged"] = best_epoch < MAX_EPOCHS - PATIENCE
            eval_results[cap_name]["splits"][split_name] = m
            logger.info(
                "  [%s][%s]  per_drug_r=%.4f  global_r=%.4f  best_epoch=%d  converged=%s",
                cap_name, split_name, m["per_drug_r"], m["global_r"], best_epoch, m["converged"],
            )

    return {"capacity_sweep": eval_results, "best_configs": best_configs}


def main(smoke: bool = False) -> None:
    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    cell_lines = overlap["depmap_id"].tolist()

    dr = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    dr = pd.DataFrame(dr[dr["depmap_id"].isin(cell_lines)].reset_index(drop=True))

    if smoke:
        dr = dr.sample(n=min(2000, len(dr)), random_state=0).reset_index(drop=True)
        logger.info("[SMOKE] Subsample: %d pairs", len(dr))

    rna_mat, rna_order = load_omics(["rna"], cell_lines)
    pair_X = build_pair_features(dr, rna_mat, rna_order)
    logger.info("Device: %s  Pairs: %d  RNA features: %d", DEVICE, len(dr), rna_mat.shape[1])

    t0 = time.perf_counter()
    result = sweep_capacity(dr, pair_X, smoke=smoke)
    logger.info("Total runtime: %.1f min", (time.perf_counter() - t0) / 60)

    if smoke:
        logger.info("[SMOKE] Pipeline OK — not writing results.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save best configs for modality ablation
    cfg_path = RESULTS_DIR / "capacity_sweep_best_configs.json"
    cfg_path.write_text(json.dumps(result["best_configs"], indent=2))
    logger.info("Best configs: %s", cfg_path)

    out = RESULTS_DIR / "capacity_sweep_results.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info("Results: %s", out)

    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REPORT_DATA.read_text()) if REPORT_DATA.exists() else {}
    existing["capacity_sweep"] = result
    REPORT_DATA.write_text(json.dumps(existing, indent=2))
    logger.info("Report data: %s", REPORT_DATA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
