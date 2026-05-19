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
ALL_CONDITIONS = ["ridge", "xgboost", "mlp"]


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

    # Merge fold_results: each per-condition file has fold_results[cname] = list of fold dicts
    merged_folds: dict[str, list[dict]] = {}
    for cname, data in per_condition.items():
        fr = data.get("fold_results", {})
        if cname in fr:
            merged_folds[cname] = fr[cname]
        elif isinstance(fr, list):
            # older format: list of fold dicts with cname key
            merged_folds[cname] = [f[cname] for f in fr if cname in f]

    # Summary
    base_mean = float(np.mean([f["per_drug_r"] for f in merged_folds.get("ridge", [])])) \
        if "ridge" in merged_folds else float("nan")

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
            "delta_vs_ridge": round(m - base_mean, 4) if not np.isnan(base_mean) else None,
        }
        logger.info("  %-8s  %.4f ± %.4f", cname, m, s)

    if "xgboost" in summary and "mlp" in summary and not np.isnan(base_mean):
        best_nonlinear = max(summary["xgboost"]["per_drug_r_mean"],
                             summary["mlp"]["per_drug_r_mean"])
        if best_nonlinear - base_mean > 0.01:
            verdict = f"Nonlinear model exceeds Ridge by Δ={best_nonlinear - base_mean:.3f} — ceiling is Ridge-limited."
        else:
            verdict = "Nonlinear models Δ≤0.01 — ceiling is not Ridge-specific; drug-blind problem is fundamentally hard."
        logger.info("Verdict: %s", verdict)
    else:
        verdict = None

    output: dict = {"summary": summary, "fold_results": merged_folds}
    if verdict:
        output["verdict"] = verdict

    out_path = report_dir / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Aggregated results → %s", out_path)


if __name__ == "__main__":
    main()
