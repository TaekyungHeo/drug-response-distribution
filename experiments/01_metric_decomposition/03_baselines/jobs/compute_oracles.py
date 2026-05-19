"""Stage 0: Analytical upper bounds for cell-only drug response prediction.

Computes:
  - Cell-mean oracle: upper bound for any drug-free model on each split
  - Global mean predictor: sanity check (should give r ≈ 0)
  - Per-drug mean predictor: confirms drug-identity signal does not contaminate metric

Oracle construction per split:
  mixed-set / drug-blind : oracle target = per-cell mean over TRAINING pairs
  cell-blind             : oracle target = per-cell mean over TEST pairs (cells unseen
                           during training; oracle uses test set itself as upper bound)

Usage:
    uv run python3 experiments/01_metric_decomposition/01_baselines/jobs/compute_oracles.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import cell_blind_split, drug_blind_split, mixed_set_split
from src.evaluation.metrics import evaluate_full

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = (
    REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "results" / "stage0"
)
REPORT_DATA = (
    REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "report" / "data" / "metrics.json"
)

SPLIT_FNS = {
    "mixed_set": mixed_set_split,
    "cell_blind": cell_blind_split,
    "drug_blind": drug_blind_split,
}


def cell_mean_preds(
    dr: pd.DataFrame,
    source_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Predict per-cell mean from source pairs; fall back to grand mean for unseen cells."""
    grand_mean = float(dr.iloc[source_idx]["ln_ic50"].mean())
    cell_mean = dr.iloc[source_idx].groupby("depmap_id")["ln_ic50"].mean()
    return (
        dr.iloc[test_idx]["depmap_id"]
        .map(cell_mean)
        .fillna(grand_mean)
        .to_numpy(dtype=np.float32)
    )


def main() -> None:
    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    dr = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    dr = dr[dr["depmap_id"].isin(overlap["depmap_id"].tolist())].reset_index(drop=True)

    print(f"Pairs: {len(dr):,}  cells: {dr['depmap_id'].nunique()}  drugs: {dr['drug_name'].nunique()}")

    results: dict = {}

    for split_name, split_fn in SPLIT_FNS.items():
        train_idx, val_idx, test_idx = split_fn(dr)
        test_df = dr.iloc[test_idx]
        y = test_df["ln_ic50"].to_numpy(dtype=np.float32)
        drugs = test_df["drug_name"].to_numpy()
        cells = test_df["depmap_id"].to_numpy()

        print(f"\n=== {split_name}  (train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}) ===")

        # -- Cell-mean oracle ---------------------------------------------------
        # Cell-blind: oracle knows the true per-cell test mean (theoretical ceiling).
        # Mixed-set / drug-blind: oracle uses training-set cell means.
        source_idx = test_idx if split_name == "cell_blind" else train_idx
        oracle_preds = cell_mean_preds(dr, source_idx, test_idx)
        oracle_m = evaluate_full(y, oracle_preds, drugs, cells)
        print(f"  cell-mean oracle  global_r={oracle_m['global_r']:.4f}"
              f"  per_drug_r={oracle_m['per_drug_r']:.4f}"
              f"  per_cell_r={oracle_m['per_cell_r']:.4f}")

        # -- Global mean predictor (sanity: should give r ≈ 0) -----------------
        grand_mean = float(dr.iloc[train_idx]["ln_ic50"].mean())
        gm_preds = np.full(len(test_idx), grand_mean, dtype=np.float32)
        gm_m = evaluate_full(y, gm_preds, drugs, cells)
        print(f"  global mean       global_r={gm_m['global_r']:.4f}"
              f"  per_drug_r={gm_m['per_drug_r']:.4f}"
              f"  per_cell_r={gm_m['per_cell_r']:.4f}")

        # -- Per-drug mean predictor (drug-identity leak check) ----------------
        drug_mean = dr.iloc[train_idx].groupby("drug_name")["ln_ic50"].mean()
        dm_preds = (
            test_df["drug_name"].map(drug_mean).fillna(grand_mean).to_numpy(dtype=np.float32)
        )
        dm_m = evaluate_full(y, dm_preds, drugs, cells)
        print(f"  per-drug mean     global_r={dm_m['global_r']:.4f}"
              f"  per_drug_r={dm_m['per_drug_r']:.4f}"
              f"  per_cell_r={dm_m['per_cell_r']:.4f}")

        results[split_name] = {
            "cell_mean_oracle": oracle_m,
            "global_mean": gm_m,
            "per_drug_mean": dm_m,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "oracle_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")

    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REPORT_DATA.read_text()) if REPORT_DATA.exists() else {}
    existing["oracles"] = results
    REPORT_DATA.write_text(json.dumps(existing, indent=2))
    print(f"Report data: {REPORT_DATA}")


if __name__ == "__main__":
    main()
