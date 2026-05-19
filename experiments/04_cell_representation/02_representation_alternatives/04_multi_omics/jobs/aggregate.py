"""Aggregate per-condition results_<condition>.json → results.json.

Run after all per-condition sbatch jobs complete:
  python aggregate.py
"""
from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).parents[1]
ALL_CONDITIONS = ["rna_mut", "rna_mut_cnv", "rna_mut_metab", "rna_mut_all"]


def main() -> None:
    report_dir = EXP_DIR / "report" / "data"

    per_condition: dict[str, dict] = {}
    for cname in ALL_CONDITIONS:
        path = report_dir / f"results_{cname}.json"
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue
        per_condition[cname] = json.loads(path.read_text())
        logger.info("Loaded: %s", path.name)

    if not per_condition:
        logger.error("No per-condition files found")
        return

    results: dict[str, dict] = {}
    baseline_mean: float | None = None
    for cname in ALL_CONDITIONS:
        if cname not in per_condition:
            continue
        data = per_condition[cname]
        # Each per-condition file is already in results format (dict keyed by condition)
        if cname in data:
            results[cname] = data[cname]
        else:
            results[cname] = data
        if cname == "rna_mut":
            baseline_mean = results[cname].get("per_drug_r_mean")

    for cname, res in results.items():
        if baseline_mean is not None:
            res["delta_vs_rna_mut"] = round(res.get("per_drug_r_mean", float("nan")) - baseline_mean, 4)

    logger.info("%-20s  %10s  %6s  %s", "Condition", "per-drug r", "±std", "delta")
    for cname, res in results.items():
        logger.info("%-20s  %10.4f  %6.4f  %+.4f",
                    cname, res.get("per_drug_r_mean", float("nan")),
                    res.get("per_drug_r_std", float("nan")),
                    res.get("delta_vs_rna_mut", 0.0))

    out_path = report_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Aggregated results → %s", out_path)


if __name__ == "__main__":
    main()
