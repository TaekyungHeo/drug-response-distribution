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
ALL_CONDITIONS = ["A_rna_mut_pca", "B_scfoundation", "C_concat"]


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
        if isinstance(fr, dict) and cname in fr:
            merged_folds[cname] = fr[cname]
        elif isinstance(fr, list):
            merged_folds[cname] = fr

    base_vals = merged_folds.get("A_rna_mut_pca", [])
    base_mean = float(np.mean([f["per_drug_r"] for f in base_vals])) if base_vals else float("nan")

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
            "delta_vs_A": round(m - base_mean, 4) if not np.isnan(base_mean) else None,
        }
        logger.info("  %-20s  per-drug r=%.4f ± %.4f", cname, m, s)

    verdict = None
    if "A_rna_mut_pca" in summary and "B_scfoundation" in summary:
        d = summary["B_scfoundation"]["delta_vs_A"] or 0.0
        if abs(d) < 0.005:
            verdict = "CONFIRMED: scFoundation converges to same per-drug r as RNA-PCA — ceiling is fundamental"
        elif d > 0.01:
            verdict = "SURPRISING: scFoundation improves per-drug r — novel biological signal"
        else:
            verdict = f"MARGINAL: Δ={d:+.4f}"
        logger.info("VERDICT: %s", verdict)

    out: dict = {"summary": summary, "fold_results": merged_folds}
    if verdict:
        out["verdict"] = verdict

    out_path = report_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Aggregated results → %s", out_path)


if __name__ == "__main__":
    main()
