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
ALL_CONDITIONS = ["A_rna_mut", "B_rppa", "C_rna_mut_rppa"]
FOCUS_MOA = [
    "Apoptosis regulation", "ERK MAPK signaling", "EGFR signaling",
    "PI3K/MTOR signaling", "Mitosis",
]


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
    merged_moa: dict[str, dict] = {}
    for cname, data in per_condition.items():
        fr = data.get("fold_results", {})
        if isinstance(fr, dict) and cname in fr:
            merged_folds[cname] = fr[cname]
        elif isinstance(fr, list):
            merged_folds[cname] = fr
        # Collect moa_fold_results
        mfr = data.get("moa_fold_results", {})
        if cname in mfr:
            merged_moa[cname] = mfr[cname]

    base_vals = merged_folds.get("A_rna_mut", [])
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

    # Apoptosis verdict
    verdict = None
    apop_a = float(np.mean(merged_moa.get("A_rna_mut", {}).get("Apoptosis regulation", []))) \
        if merged_moa.get("A_rna_mut", {}).get("Apoptosis regulation") else float("nan")
    apop_b = float(np.mean(merged_moa.get("B_rppa", {}).get("Apoptosis regulation", []))) \
        if merged_moa.get("B_rppa", {}).get("Apoptosis regulation") else float("nan")
    apop_c = float(np.mean(merged_moa.get("C_rna_mut_rppa", {}).get("Apoptosis regulation", []))) \
        if merged_moa.get("C_rna_mut_rppa", {}).get("Apoptosis regulation") else float("nan")
    max_rppa = max(apop_b, apop_c) if not (np.isnan(apop_b) or np.isnan(apop_c)) else float("nan")
    if not np.isnan(max_rppa) and not np.isnan(apop_a):
        if max_rppa - apop_a > 0.05:
            verdict = "RPPA breaks Apoptosis ceiling — revise 'genuine biological limit'"
        elif abs(max_rppa - apop_a) <= 0.02:
            verdict = "RPPA comparable to RNA — 'genuine biological limit' claim strengthened"
        else:
            verdict = f"Marginal RPPA effect (Δ={max_rppa - apop_a:+.3f})"
        logger.info("VERDICT: %s", verdict)

    out: dict = {
        "summary": summary,
        "fold_results": merged_folds,
        "moa_fold_results": merged_moa,
    }
    if verdict:
        out["apoptosis_verdict"] = verdict

    out_path = report_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Aggregated results → %s", out_path)


if __name__ == "__main__":
    main()
