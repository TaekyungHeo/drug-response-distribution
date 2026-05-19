"""Step 2: Per-drug metrics and bootstrap CI for a single fold.

Called once per fold (0-4) via SLURM array or --fold argument.
Reads fold predictions from results/fold_predictions/ridge_drug_blind_fold{i}.parquet.
Writes per-fold intermediate results to results/fold_metrics/.

Runtime per fold: ~1-2 min CPU.
Output:
  results/fold_metrics/fold{i}_per_drug.parquet   — per-drug metric values
  results/fold_metrics/fold{i}_bootstrap.json     — bootstrap CI widths
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[4]
EXP_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

PRED_DIR = EXP_DIR / "results" / "fold_predictions"
OUT_DIR = EXP_DIR / "results" / "fold_metrics"
N_BOOT = 200
MIN_CELLS = 5
METRIC_NAMES = ["r_p", "r_s", "tau", "ndcg5", "r2"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_per_drug(fold_df: pd.DataFrame, fold_i: int) -> pd.DataFrame:
    from src.evaluation.per_drug_metrics import compute_all

    rows = []
    for drug, grp in fold_df.groupby("drug_name"):
        y_true = grp["y_true"].to_numpy(dtype=np.float64)
        y_pred = grp["y_pred"].to_numpy(dtype=np.float64)
        if len(y_true) < MIN_CELLS or np.std(y_true) < 1e-6:
            continue

        metrics = compute_all(y_true, y_pred)
        y_2a = np.full_like(y_true, y_true.mean())
        metrics_2a = compute_all(y_true, y_2a)

        row: dict = {"fold": fold_i, "drug_name": drug, "n_cells": len(y_true)}
        row.update(metrics)
        row.update({f"{k}_2a": v for k, v in metrics_2a.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def compute_bootstrap(fold_df: pd.DataFrame) -> dict:
    from src.evaluation.per_drug_metrics import (
        bootstrap_ci_width, kendall_tau, ndcg_at_5,
        pearson_r, r2_drug_mean, spearman_r,
    )

    metric_fns = {
        "r_p": pearson_r, "r_s": spearman_r, "tau": kendall_tau,
        "ndcg5": ndcg_at_5, "r2": r2_drug_mean,
    }
    ci_widths: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}

    for drug, grp in fold_df.groupby("drug_name"):
        y_true = grp["y_true"].to_numpy(dtype=np.float64)
        y_pred = grp["y_pred"].to_numpy(dtype=np.float64)
        if len(y_true) < MIN_CELLS or np.std(y_true) < 1e-6:
            continue
        for m, fn in metric_fns.items():
            w = bootstrap_ci_width(
                y_true, y_pred, fn,
                n_boot=N_BOOT,
                seed=hash(drug) % (2**31),
            )
            if not np.isnan(w):
                ci_widths[m].append(w)

    return {m: vals for m, vals in ci_widths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Fold index 0-4. Defaults to SLURM_ARRAY_TASK_ID if set.",
    )
    args = parser.parse_args()

    fold_i = args.fold
    if fold_i is None:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is not None:
            fold_i = int(task_id)
    if fold_i is None:
        raise ValueError("Specify --fold 0-4 or set SLURM_ARRAY_TASK_ID.")

    pred_path = PRED_DIR / f"ridge_drug_blind_fold{fold_i}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} — run save_ridge_predictions.py --fold {fold_i} first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_df = pd.read_parquet(pred_path)
    log.info("Fold %d: %d test pairs, %d drugs",
             fold_i, len(fold_df), fold_df["drug_name"].nunique())

    per_drug_df = compute_per_drug(fold_df, fold_i)
    per_drug_df.to_parquet(OUT_DIR / f"fold{fold_i}_per_drug.parquet", index=False)
    log.info("Fold %d: %d (drug, fold) entries", fold_i, len(per_drug_df))

    ci_widths = compute_bootstrap(fold_df)
    with (OUT_DIR / f"fold{fold_i}_bootstrap.json").open("w") as f:
        json.dump(ci_widths, f, indent=2)
    log.info("Fold %d: bootstrap CI widths saved.", fold_i)


if __name__ == "__main__":
    main()
