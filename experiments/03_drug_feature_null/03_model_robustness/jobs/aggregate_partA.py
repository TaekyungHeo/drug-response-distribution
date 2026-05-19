"""Aggregate per-fold shard results from partA into partA_metrics.json.

Usage:
    python experiments/03_drug_feature_null/03_model_robustness/jobs/aggregate_partA.py
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
CONDITIONS = ["morgan_fp", "no_drug"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    report_dir = EXP_DIR / "report" / "data"
    shard_paths = sorted(report_dir.glob("fold_??_partA_results.json"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No partA shard files found in {report_dir}. "
            "Run: sbatch sbatch_partA_array.sh"
        )

    logger.info("Found %d shard files", len(shard_paths))
    missing = [k for k in range(K_FOLDS) if not (report_dir / f"fold_{k:02d}_partA_results.json").exists()]
    if missing:
        raise RuntimeError(f"Missing partA shards for folds: {missing}")

    all_results = {c: [] for c in CONDITIONS}
    for p in shard_paths:
        with p.open() as f:
            shard = json.load(f)
        for c in CONDITIONS:
            all_results[c].extend(shard["results"].get(c, []))

    output = {}
    for condition in CONDITIONS:
        fold_rs = [r["test_per_drug_r"] for r in all_results[condition]]
        valid = [r for r in fold_rs if not np.isnan(r)]
        output[condition] = {
            "mean": float(np.mean(valid)) if valid else float("nan"),
            "std": float(np.std(valid)) if valid else float("nan"),
            "folds": fold_rs,
        }

    delta = output["morgan_fp"]["mean"] - output["no_drug"]["mean"]
    output["delta_morgan_vs_no_drug"] = float(delta)

    metrics_path = report_dir / "partA_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("morgan_fp : mean=%.4f ± %.4f  folds=%s",
                output["morgan_fp"]["mean"], output["morgan_fp"]["std"],
                " ".join(f"{r:.4f}" for r in output["morgan_fp"]["folds"]))
    logger.info("no_drug   : mean=%.4f ± %.4f  folds=%s",
                output["no_drug"]["mean"], output["no_drug"]["std"],
                " ".join(f"{r:.4f}" for r in output["no_drug"]["folds"]))
    logger.info("delta_morgan_vs_no_drug: %.4f", delta)
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
