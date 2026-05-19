"""Aggregate per-fold shard results from partB into gnn_embeddings_256.npy + partB_metrics.json.

Reads:
  report/data/fold_??_partB_results.json   — fold metrics
  report/data/fold_??_embeddings.npz       — drug_indices + embeddings arrays

Writes:
  data/processed/gnn_embeddings_256.npy    — full embedding matrix (n_drugs × 256)
  report/data/partB_metrics.json           — aggregated metrics

Usage:
    python experiments/03_drug_feature_null/03_model_robustness/jobs/aggregate_partB.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
K_FOLDS = 10
D_MODEL = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    report_dir = EXP_DIR / "report" / "data"

    # Verify all shards present
    missing_results = [k for k in range(K_FOLDS)
                       if not (report_dir / f"fold_{k:02d}_partB_results.json").exists()]
    missing_embs = [k for k in range(K_FOLDS)
                    if not (report_dir / f"fold_{k:02d}_embeddings.npz").exists()]
    if missing_results:
        raise RuntimeError(f"Missing partB result shards for folds: {missing_results}")
    if missing_embs:
        raise RuntimeError(f"Missing embedding shards for folds: {missing_embs}")

    logger.info("All %d shard pairs found.", K_FOLDS)

    # Determine total drug count from embedding shards
    all_drug_indices: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    fold_results = []

    for k in range(K_FOLDS):
        npz = np.load(report_dir / f"fold_{k:02d}_embeddings.npz")
        d_idx = npz["drug_indices"]
        embs = npz["embeddings"]
        logger.info("  fold %02d: %d test drugs embedded", k, len(d_idx))
        all_drug_indices.append(d_idx)
        all_embeddings.append(embs)

        with open(report_dir / f"fold_{k:02d}_partB_results.json") as f:
            shard = json.load(f)
        fold_results.append(shard["results"])

    # Build full embedding matrix
    all_idx_cat = np.concatenate(all_drug_indices)
    n_drugs = int(all_idx_cat.max()) + 1
    gnn_embeddings = np.full((n_drugs, D_MODEL), np.nan, dtype=np.float32)

    for d_idx_arr, embs in zip(all_drug_indices, all_embeddings):
        for drug_idx_val, emb in zip(d_idx_arr, embs):
            gnn_embeddings[int(drug_idx_val)] = emb

    # Coverage check
    missing = [int(i) for i in range(n_drugs) if np.any(np.isnan(gnn_embeddings[i]))]
    if missing:
        raise RuntimeError(
            f"Coverage FAILED: {len(missing)} drug indices have NaN embeddings: {missing[:10]}"
        )
    logger.info("Coverage check PASSED: all %d drugs embedded.", n_drugs)

    # Save embeddings
    emb_path = DATA_DIR / "gnn_embeddings_256.npy"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, gnn_embeddings)
    logger.info("GNN embeddings saved to %s  shape=%s", emb_path, gnn_embeddings.shape)

    # Aggregate metrics
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
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
