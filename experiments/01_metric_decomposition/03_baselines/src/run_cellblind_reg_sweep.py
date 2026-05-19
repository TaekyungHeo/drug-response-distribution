"""Stage 3: Cell-blind regularization sweep.

Cell-blind val per-drug r peaks early then declines — a sign of overfitting to
training cell lines. This stage asks: is overfitting the binding constraint, or
is the bottleneck the data itself?

Protocol:
  - RNA-only MLP, fixed architecture [512→256→64→1]
  - Grid: dropout [0.2, 0.3, 0.5] × weight_decay [1e-3, 1e-2, 5e-2]
  - Training: LR cosine [1e-3→1e-5], max_epochs=300, early stopping patience=30
  - Val metric: per-drug r (primary)
  - Reports: best_val_per_drug_r, best_epoch, val curve per config

Convergence check: if best_epoch >= 270 (= 300 - 30), flag run as potentially
unconverged — re-run with more epochs before drawing conclusions.

If no config improves beyond the Stage 2 cell-blind per-drug r →
regularization is not the binding constraint; bottleneck is the data.

Usage:
    uv run python3 experiments/01_metric_decomposition/03_baselines/src/run_cellblind_reg_sweep.py
    uv run python3 experiments/01_metric_decomposition/03_baselines/src/run_cellblind_reg_sweep.py --smoke
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
from src.data.splits import cell_blind_split
from src.evaluation.metrics import evaluate_full
from src.train import predict_batched, train_mlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "experiments" / "01_metric_decomposition" / "03_baselines" / "results" / "stage3"
REPORT_DATA = REPO_ROOT / "experiments" / "01_metric_decomposition" / "03_baselines" / "report" / "data" / "metrics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096 if DEVICE == "cuda" else 2048
MAX_EPOCHS = 300
PATIENCE = 30
HIDDEN_DIMS = [512, 256, 64]

DROPOUT_GRID = [0.2, 0.3, 0.5]
WD_GRID = [1e-3, 1e-2, 5e-2]


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

    train_idx, val_idx, test_idx = cell_blind_split(dr, seed=42)
    Xtr, Xvl, Xte = z_score_normalize(pair_X[train_idx], pair_X[val_idx], pair_X[test_idx])
    ytr = dr.iloc[train_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    yvl = dr.iloc[val_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    val_drugs = dr.iloc[val_idx]["drug_name"].to_numpy()
    y_te = dr.iloc[test_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    drugs_te = dr.iloc[test_idx]["drug_name"].to_numpy()
    cells_te = dr.iloc[test_idx]["depmap_id"].to_numpy()
    logger.info("Split: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    dropouts = DROPOUT_GRID[:1] if smoke else DROPOUT_GRID
    wds = WD_GRID[:1] if smoke else WD_GRID
    max_ep = 2 if smoke else MAX_EPOCHS

    t0 = time.perf_counter()
    grid_results = []

    for dropout in dropouts:
        for wd in wds:
            logger.info("  dropout=%.1f  weight_decay=%g", dropout, wd)
            model, curve, best_val_r, best_epoch = train_mlp(
                Xtr, ytr, Xvl, yvl, val_drugs,
                HIDDEN_DIMS, dropout, wd, DEVICE,
                max_epochs=max_ep, patience=PATIENCE, batch_size=BATCH_SIZE,
            )
            test_preds = predict_batched(model, Xte, DEVICE, BATCH_SIZE)
            test_m = dict(evaluate_full(y_te, test_preds, drugs_te, cells_te))
            converged = best_epoch < MAX_EPOCHS - PATIENCE
            entry = {
                "dropout": dropout,
                "weight_decay": wd,
                "best_val_per_drug_r": best_val_r,
                "best_epoch": best_epoch,
                "converged": converged,
                "val_per_drug_r_curve": [round(v, 5) if not np.isnan(v) else None for v in curve],
                **{k: v for k, v in test_m.items() if k != "n"},
                "n": test_m["n"],
            }
            grid_results.append(entry)
            logger.info(
                "    best_val_per_drug_r=%.4f @ epoch %d  converged=%s"
                "  test_per_drug_r=%.4f  test_global_r=%.4f",
                best_val_r, best_epoch, converged, test_m["per_drug_r"], test_m["global_r"],
            )

    logger.info("Total runtime: %.1f min", (time.perf_counter() - t0) / 60)

    best = max(grid_results, key=lambda x: x["best_val_per_drug_r"])
    logger.info(
        "Best config: dropout=%.1f wd=%g  best_val_per_drug_r=%.4f @ epoch %d  converged=%s",
        best["dropout"], best["weight_decay"], best["best_val_per_drug_r"],
        best["best_epoch"], best["converged"],
    )

    if smoke:
        logger.info("[SMOKE] Pipeline OK — not writing results.")
        return

    results = {
        "grid": grid_results,
        "best": {k: v for k, v in best.items() if k != "val_per_drug_r_curve"},
        "architecture": HIDDEN_DIMS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "convergence_threshold": MAX_EPOCHS - PATIENCE,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "cellblind_reg_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("Results: %s", out)

    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REPORT_DATA.read_text()) if REPORT_DATA.exists() else {}
    existing["cellblind_reg_sweep"] = results
    REPORT_DATA.write_text(json.dumps(existing, indent=2))
    logger.info("Report data: %s", REPORT_DATA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
