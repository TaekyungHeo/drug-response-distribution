"""Stage 1: Ridge regression — architecture-free cell-only baseline.

Answers: is global r ≈ 0.32 a data limitation or an MLP architecture limitation?
  Ridge >= MLP  →  MLP was undertrained or under-regularized
  Ridge ≈  MLP  →  linear model at ceiling; data is the bottleneck
  Ridge <  MLP  →  nonlinearity genuinely helps

Protocol:
  - Input: RNA-seq only (19,193 features), z-score normalised on train set
  - Alpha: [0.01, 0.1, 1, 10, 100, 1000] selected by best val-set global Pearson r
  - 5-fold CV (different seeds) on each of the three split protocols
  - Metrics: global r, per-drug r, per-cell r

Usage:
    uv run python3 experiments/01_metric_decomposition/01_baselines/jobs/run_ridge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.data.omics_utils import build_pair_features, load_omics, z_score_normalize
from src.data.splits import cell_blind_split, drug_blind_split, mixed_set_split
from src.evaluation.metrics import evaluate_full

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = (
    REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "results" / "stage1"
)
REPORT_DATA = (
    REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "report" / "data" / "metrics.json"
)

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_FOLDS = 5

SPLIT_FNS = {
    "mixed_set": mixed_set_split,
    "cell_blind": cell_blind_split,
    "drug_blind": drug_blind_split,
}


def _global_r(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.std() < 1e-9:
        return float("nan")
    return float(pearsonr(preds.astype(np.float64), targets.astype(np.float64))[0])  # type: ignore[arg-type]


def run_one_fold(
    pair_X: np.ndarray,
    dr: pd.DataFrame,
    split_fn,
    seed: int,
) -> dict:
    train_idx, val_idx, test_idx = split_fn(dr, seed=seed)

    Xtr, Xvl, Xte = z_score_normalize(pair_X[train_idx], pair_X[val_idx], pair_X[test_idx])
    ytr = dr.iloc[train_idx]["ln_ic50"].to_numpy(dtype=np.float64)
    yvl = dr.iloc[val_idx]["ln_ic50"].to_numpy(dtype=np.float64)

    # Select alpha by val global r
    best_alpha, best_val_r = ALPHA_GRID[0], -np.inf
    for alpha in ALPHA_GRID:
        m = Ridge(alpha=alpha)
        m.fit(Xtr, ytr)
        vr = _global_r(m.predict(Xvl), yvl)
        if vr > best_val_r:
            best_val_r, best_alpha = vr, alpha

    # Refit with best alpha and evaluate on test
    m = Ridge(alpha=best_alpha)
    m.fit(Xtr, ytr)
    test_preds = m.predict(Xte).astype(np.float32)
    y_test = dr.iloc[test_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    drugs = dr.iloc[test_idx]["drug_name"].to_numpy()
    cells = dr.iloc[test_idx]["depmap_id"].to_numpy()

    metrics = dict(evaluate_full(y_test, test_preds, drugs, cells))
    metrics["best_alpha"] = best_alpha
    metrics["val_r"] = float(best_val_r)
    metrics["seed"] = seed
    return metrics


def main() -> None:
    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    cell_lines = overlap["depmap_id"].tolist()

    dr = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    dr = pd.DataFrame(dr[dr["depmap_id"].isin(cell_lines)].reset_index(drop=True))

    rna_mat, rna_order = load_omics(["rna"], cell_lines)
    pair_X = build_pair_features(dr, rna_mat, rna_order).astype(np.float64)

    print(f"Pairs: {len(dr):,}  RNA features: {rna_mat.shape[1]}")

    results: dict = {}

    for split_name, split_fn in SPLIT_FNS.items():
        print(f"\n=== {split_name} ===")
        folds = []

        for seed in range(N_FOLDS):
            m = run_one_fold(pair_X, dr, split_fn, seed)
            folds.append(m)
            print(
                f"  seed={seed}  alpha={m['best_alpha']:7.2f}  val_r={m['val_r']:.4f}"
                f"  global_r={m['global_r']:.4f}"
                f"  per_drug_r={m['per_drug_r']:.4f}"
                f"  per_cell_r={m['per_cell_r']:.4f}"
            )

        gr = [f["global_r"] for f in folds]
        print(
            f"  → global_r={np.mean(gr):.4f} ± {np.std(gr):.4f}"
            f"  per_drug_r={np.mean([f['per_drug_r'] for f in folds]):.4f}"
            f"  per_cell_r={np.mean([f['per_cell_r'] for f in folds]):.4f}"
        )

        results[split_name] = {
            "global_r_mean": float(np.mean(gr)),
            "global_r_std": float(np.std(gr)),
            "per_drug_r_mean": float(np.mean([f["per_drug_r"] for f in folds])),
            "per_cell_r_mean": float(np.mean([f["per_cell_r"] for f in folds])),
            "folds": folds,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ridge_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")

    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REPORT_DATA.read_text()) if REPORT_DATA.exists() else {}
    existing["ridge"] = results
    REPORT_DATA.write_text(json.dumps(existing, indent=2))
    print(f"Report data: {REPORT_DATA}")


if __name__ == "__main__":
    main()
