"""Aggregate per-fold shard results from 06_objective_axis into metrics.json.

Usage (after all array jobs complete):
    python experiments/03_drug_feature_null/06_objective_axis/jobs/aggregate.py
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
K_FOLDS = 10
CONDITIONS = ["mlp_mse_no_drug", "mlp_mse_morgan", "mlp_ranknet_morgan"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    report_dir = EXP_DIR / "report" / "data"
    shard_paths = sorted(report_dir.glob("fold_??_results.json"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No shard files found in {report_dir}. "
            "Run the array job first: sbatch sbatch_array.sh"
        )

    logger.info("Found %d shard files: %s", len(shard_paths), [p.name for p in shard_paths])
    missing = [k for k in range(K_FOLDS) if not (report_dir / f"fold_{k:02d}_results.json").exists()]
    if missing:
        raise RuntimeError(f"Missing shards for folds: {missing}. Cannot aggregate.")

    all_fold_results = []
    for p in shard_paths:
        with p.open() as f:
            all_fold_results.extend(json.load(f))

    logger.info("Total fold-condition results loaded: %d", len(all_fold_results))

    results_by_condition = {}
    for cond in CONDITIONS:
        fold_rs = [r["test_per_drug_r"] for r in all_fold_results if r["condition"] == cond]
        if len(fold_rs) != K_FOLDS:
            logger.warning("Condition %s: expected %d folds, got %d", cond, K_FOLDS, len(fold_rs))
        results_by_condition[cond] = {
            "mean": float(np.mean(fold_rs)),
            "std": float(np.std(fold_rs)),
            "folds": [float(r) for r in fold_rs],
        }

    base_mean = results_by_condition["mlp_mse_no_drug"]["mean"]
    for cond in ("mlp_mse_morgan", "mlp_ranknet_morgan"):
        results_by_condition[cond]["delta"] = float(
            results_by_condition[cond]["mean"] - base_mean
        )

    ranknet_vs_mse_delta = float(
        results_by_condition["mlp_ranknet_morgan"]["mean"]
        - results_by_condition["mlp_mse_morgan"]["mean"]
    )
    results_by_condition["ranknet_vs_mse_delta"] = ranknet_vs_mse_delta  # type: ignore[assignment]

    metrics_path = report_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(results_by_condition, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for cond in CONDITIONS:
        r = results_by_condition[cond]
        delta_str = f"  Δ={r.get('delta', 0.0):+.4f}" if "delta" in r else ""
        logger.info(
            "  %-22s  mean=%.4f ± %.4f%s  folds=%s",
            cond, r["mean"], r["std"], delta_str,
            " ".join(f"{v:.3f}" for v in r["folds"]),
        )
    logger.info("  ranknet_vs_mse_delta:  %+.4f", ranknet_vs_mse_delta)
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
