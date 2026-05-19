"""Stage 1: Ridge regression — architecture-free cell-only baseline.

Answers: is the RNA-only baseline a data limitation or an MLP architecture limitation?
  Ridge >= MLP  →  MLP was the bottleneck (undertrained or under-regularized)
  Ridge ≈  MLP  →  linear model at ceiling; data is the bottleneck
  Ridge <  MLP  →  nonlinearity genuinely helps

Protocol:
  - Input: RNA-seq only (19,193 features), z-score normalised on train set
  - Alpha: [0.01, 0.1, 1, 10, 100, 1000] selected by best val-set per-drug r
  - 5-fold CV (seeds 0–4) on each of the three split protocols
  - Metrics reported: per-drug r (primary), global r (secondary), per-cell r

Usage:
    uv run python3 experiments/01_metric_decomposition/03_baselines/src/run_ridge.py
    uv run python3 experiments/01_metric_decomposition/03_baselines/src/run_ridge.py --smoke
"""

from __future__ import annotations

import argparse
import json
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
from src.evaluation.per_drug import mean_per_drug_r

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "experiments" / "01_metric_decomposition" / "03_baselines" / "results" / "stage1"
REPORT_DATA = REPO_ROOT / "experiments" / "01_metric_decomposition" / "03_baselines" / "report" / "data" / "metrics.json"

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_FOLDS = 5

SPLIT_FNS = {
    "mixed_set": mixed_set_split,
    "cell_blind": cell_blind_split,
    "drug_blind": drug_blind_split,
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


class _GpuRidgeCache:
    """Precomputed Gram matrix for multi-alpha Ridge solve on GPU."""

    def __init__(self, Xtr: np.ndarray, ytr: np.ndarray) -> None:
        Xtr_gpu = torch.tensor(Xtr, dtype=DTYPE, device=DEVICE)
        ytr_gpu = torch.tensor(ytr, dtype=DTYPE, device=DEVICE)

        self.X_mean = Xtr_gpu.mean(0)
        self.y_mean = ytr_gpu.mean().item()
        Xc = Xtr_gpu - self.X_mean
        yc = ytr_gpu - self.y_mean

        n, d = Xc.shape
        self.use_dual = n < d

        if self.use_dual:
            self.K = Xc @ Xc.T  # n×n
            self.Xc = Xc
            self.yc = yc
        else:
            self.XtX = Xc.T @ Xc  # d×d
            self.Xty = Xc.T @ yc
            del Xc, yc

        del Xtr_gpu, ytr_gpu
        torch.cuda.empty_cache()

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        X_gpu = torch.tensor(X, dtype=DTYPE, device=DEVICE)

        if self.use_dual:
            n = self.K.shape[0]
            A = self.K + alpha * torch.eye(n, device=DEVICE, dtype=DTYPE)
            dual = torch.linalg.solve(A, self.yc)
            w = self.Xc.T @ dual
        else:
            d = self.XtX.shape[0]
            A = self.XtX + alpha * torch.eye(d, device=DEVICE, dtype=DTYPE)
            w = torch.linalg.solve(A, self.Xty)

        intercept = self.y_mean - self.X_mean @ w
        preds = (X_gpu @ w + intercept).cpu().numpy().astype(np.float32)
        return preds


def run_one_fold(
    pair_X: np.ndarray,
    dr: pd.DataFrame,
    split_fn,
    seed: int,
    smoke: bool = False,
) -> dict:
    train_idx, val_idx, test_idx = split_fn(dr, seed=seed)

    Xtr, Xvl, Xte = z_score_normalize(pair_X[train_idx], pair_X[val_idx], pair_X[test_idx])
    ytr = dr.iloc[train_idx]["ln_ic50"].to_numpy(dtype=np.float64)
    yvl = dr.iloc[val_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    val_drugs = dr.iloc[val_idx]["drug_name"].to_numpy()

    t0 = time.perf_counter()
    cache = _GpuRidgeCache(Xtr, ytr)
    gram_elapsed = time.perf_counter() - t0
    form = "dual" if cache.use_dual else "primal"
    print(f"    [{time.strftime('%H:%M:%S')}] Gram ({form}): {gram_elapsed:.1f}s", flush=True)

    alphas = ALPHA_GRID[:2] if smoke else ALPHA_GRID

    best_alpha, best_val_r = alphas[0], -np.inf
    for alpha in alphas:
        t0 = time.perf_counter()
        val_preds = cache.predict(Xvl, alpha)
        elapsed = time.perf_counter() - t0
        pdr = mean_per_drug_r(val_preds, yvl.astype(np.float32), val_drugs)
        print(f"    [{time.strftime('%H:%M:%S')}] alpha={alpha:8.2f}  solve={elapsed:.1f}s  val_pdr={pdr:.4f}", flush=True)
        if not np.isnan(pdr) and pdr > best_val_r:
            best_val_r, best_alpha = pdr, alpha

    test_preds = cache.predict(Xte, best_alpha)
    y_test = dr.iloc[test_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    drugs = dr.iloc[test_idx]["drug_name"].to_numpy()
    cells = dr.iloc[test_idx]["depmap_id"].to_numpy()

    metrics = dict(evaluate_full(y_test, test_preds, drugs, cells))
    metrics["best_alpha"] = best_alpha
    metrics["val_per_drug_r"] = float(best_val_r)
    metrics["seed"] = seed
    return metrics


def main(smoke: bool = False) -> None:
    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    cell_lines = overlap["depmap_id"].tolist()

    dr = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    dr = pd.DataFrame(dr[dr["depmap_id"].isin(cell_lines)].reset_index(drop=True))

    if smoke:
        dr = dr.sample(n=min(2000, len(dr)), random_state=0).reset_index(drop=True)
        print(f"[SMOKE] Subsample: {len(dr)} pairs")

    rna_mat, rna_order = load_omics(["rna"], cell_lines)
    pair_X = build_pair_features(dr, rna_mat, rna_order)  # float32 from load_omics
    print(f"Pairs: {len(dr):,}  RNA features: {rna_mat.shape[1]}")

    results: dict = {}
    n_folds = 1 if smoke else N_FOLDS

    for split_name, split_fn in SPLIT_FNS.items():
        print(f"\n[{time.strftime('%H:%M:%S')}] === {split_name} ===")
        folds = []

        for seed in range(n_folds):
            print(f"  [{time.strftime('%H:%M:%S')}] seed={seed} starting...")
            m = run_one_fold(pair_X, dr, split_fn, seed, smoke=smoke)
            folds.append(m)
            print(
                f"  [{time.strftime('%H:%M:%S')}] seed={seed}  alpha={m['best_alpha']:7.2f}"
                f"  val_per_drug_r={m['val_per_drug_r']:.4f}"
                f"  per_drug_r={m['per_drug_r']:.4f}  global_r={m['global_r']:.4f}"
            )

        pdr_vals = [f["per_drug_r"] for f in folds]
        print(
            f"  [{time.strftime('%H:%M:%S')}] → per_drug_r={np.mean(pdr_vals):.4f} ± {np.std(pdr_vals):.4f}"
            f"  global_r={np.mean([f['global_r'] for f in folds]):.4f}"
        )

        results[split_name] = {
            "per_drug_r_mean": float(np.mean(pdr_vals)),
            "per_drug_r_std": float(np.std(pdr_vals)),
            "global_r_mean": float(np.mean([f["global_r"] for f in folds])),
            "global_r_std": float(np.std([f["global_r"] for f in folds])),
            "per_cell_r_mean": float(np.mean([f["per_cell_r"] for f in folds])),
            "folds": folds,
        }

    if smoke:
        print("\n[SMOKE] Pipeline OK — not writing results.")
        return

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
