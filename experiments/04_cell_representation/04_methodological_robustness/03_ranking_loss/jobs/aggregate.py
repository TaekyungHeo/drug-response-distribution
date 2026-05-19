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
ALL_CONDITIONS = ["ridge_mse", "ridge_rank", "ridge_rank_01", "ridge_rank_10"]


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

    merged_folds: dict[str, list[dict]] = {}
    for cname, data in per_condition.items():
        fr = data.get("fold_results", {})
        if cname in fr:
            merged_folds[cname] = fr[cname]
        elif isinstance(fr, list):
            merged_folds[cname] = [f[cname] for f in fr if cname in f]

    mse_mean = float(np.mean([f["per_drug_r"] for f in merged_folds.get("ridge_mse", [])])) \
        if "ridge_mse" in merged_folds else float("nan")

    summary: dict[str, dict] = {}
    for cname in ALL_CONDITIONS:
        if cname not in merged_folds or not merged_folds[cname]:
            continue
        vals = [f["per_drug_r"] for f in merged_folds[cname]]
        m = float(np.mean(vals))
        s = float(np.std(vals))
        summary[cname] = {
            "per_drug_r_mean": round(m, 4),
            "per_drug_r_std": round(s, 4),
            "delta": round(m - mse_mean, 4) if not np.isnan(mse_mean) else None,
        }
        logger.info("  %-20s  %.4f ± %.4f", cname, m, s)

    verdict = None
    if "ridge_mse" in summary and "ridge_rank" in summary:
        rank_delta = summary["ridge_rank"]["delta"] or 0.0
        if abs(rank_delta) <= 0.005:
            verdict = (
                f"Drug-standardized Δ={rank_delta:.4f} ≤ 0.005 — "
                "MSE and ranking objectives are equivalent; ceiling is not MSE-loss-specific."
            )
        else:
            verdict = (
                f"Drug-standardized Δ={rank_delta:.4f} > 0.005 — "
                "ranking loss changes per-drug r; revise to 'MSE-loss ceiling'."
            )
        logger.info("Verdict: %s", verdict)

    output: dict = {"summary": summary, "fold_results": merged_folds}
    if verdict:
        output["verdict"] = verdict

    out_path = report_dir / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Aggregated results → %s", out_path)


if __name__ == "__main__":
    main()
