"""Aggregate Part C fold shards into partC_metrics.json.

Usage:
    uv run python3 experiments/03_drug_feature_null/03_model_robustness/jobs/aggregate_partC.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).parents[1]
REPORT_DIR = EXP_DIR / "report" / "data"
CONDITIONS = ["lincs_pca64", "drug_target", "moa_onehot"]
K_FOLDS = 10


def main() -> None:
    # Collect shards
    shards = {}
    for k in range(K_FOLDS):
        shard_path = REPORT_DIR / f"fold_{k:02d}_partC_results.json"
        if shard_path.exists():
            with open(shard_path) as f:
                shards[k] = json.load(f)
            logger.info("Loaded shard fold %d", k)
        else:
            logger.warning("Missing shard: %s", shard_path)

    if not shards:
        logger.error("No shards found!")
        return

    logger.info("Found %d/%d fold shards", len(shards), K_FOLDS)

    # Aggregate per condition
    output = {}
    for condition in CONDITIONS:
        fold_rs = []
        for k in sorted(shards.keys()):
            results = shards[k].get("results", {}).get(condition, [])
            if results:
                fold_rs.append(results[0]["test_per_drug_r"])

        if fold_rs:
            output[condition] = {
                "mean": float(np.mean(fold_rs)),
                "std": float(np.std(fold_rs)),
                "n_folds": len(fold_rs),
                "folds": fold_rs,
            }
        else:
            output[condition] = {"mean": float("nan"), "std": float("nan"), "n_folds": 0, "folds": []}

    # Load Part A baseline
    parta_path = REPORT_DIR / "partA_metrics.json"
    if parta_path.exists():
        with open(parta_path) as f:
            parta = json.load(f)
        baseline = parta.get("no_drug", {}).get("mean", 0.645)
    else:
        # Compute from Part A shards
        parta_rs = []
        for k in range(K_FOLDS):
            p = REPORT_DIR / f"fold_{k:02d}_partA_results.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                parta_rs.append(d["results"]["no_drug"][0]["test_per_drug_r"])
        baseline = float(np.mean(parta_rs)) if parta_rs else 0.645
        logger.info("Computed baseline from Part A shards: %.4f (%d folds)", baseline, len(parta_rs))

    for condition in CONDITIONS:
        if output[condition]["n_folds"] > 0:
            output[condition]["delta_vs_no_drug"] = output[condition]["mean"] - baseline

    # Write
    metrics_path = REPORT_DIR / "partC_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    logger.info("=" * 70)
    logger.info("FINAL RESULTS (Part C)")
    logger.info("-" * 70)
    logger.info("%-15s  %8s  %8s  %8s  %5s", "Condition", "Mean r", "Std", "Δ no_drug", "Folds")
    logger.info("-" * 70)
    for condition in CONDITIONS:
        m = output[condition]
        if m["n_folds"] > 0:
            logger.info("%-15s  %8.4f  %8.4f  %+8.4f  %5d",
                        condition, m["mean"], m["std"], m.get("delta_vs_no_drug", 0), m["n_folds"])
        else:
            logger.info("%-15s  %8s  %8s  %8s  %5d", condition, "—", "—", "—", 0)
    logger.info("-" * 70)
    logger.info("%-15s  %8.4f", "no_drug (ref)", baseline)
    logger.info("=" * 70)
    logger.info("Written: %s", metrics_path)


if __name__ == "__main__":
    main()
