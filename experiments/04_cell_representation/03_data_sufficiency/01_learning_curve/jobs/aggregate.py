"""Aggregate per-fraction results_frac_<f>.json → results.json.

Run after all per-fraction sbatch jobs complete:
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
ALL_FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def fraction_key(frac: float) -> str:
    return f"frac_{frac:.1f}".replace(".", "_")


def main() -> None:
    report_dir = EXP_DIR / "report" / "data"

    results: dict[str, dict] = {}
    for frac in ALL_FRACTIONS:
        key = fraction_key(frac)
        path = report_dir / f"results_{key}.json"
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue
        data = json.loads(path.read_text())
        # per-fraction file has the fraction result under the key
        if key in data:
            results[key] = data[key]
        else:
            results[key] = data
        logger.info("Loaded: %s  per_drug_r=%.4f", path.name,
                    results[key].get("per_drug_r_mean", float("nan")))

    if not results:
        logger.error("No per-fraction files found")
        return

    logger.info("%-12s  %10s  %6s  %10s", "Fraction", "per-drug r", "±std", "n_cells")
    for key, res in results.items():
        logger.info("%-12s  %10.4f  %6.4f  %10.1f",
                    f"{res.get('fraction', '?'):.1f}",
                    res.get("per_drug_r_mean", float("nan")),
                    res.get("per_drug_r_std", float("nan")),
                    res.get("n_train_cells_mean", float("nan")))

    out_path = report_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Aggregated results → %s", out_path)


if __name__ == "__main__":
    main()
