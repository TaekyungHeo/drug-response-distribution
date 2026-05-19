"""Step 2: Baseline dissociation — entry point.

Logic lives in src/analysis.py.
Runtime: 2A fast (< 2 min, no GPU). 2B requires run_model_comparison first.
Output:  results/baseline_dissociation.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parents[4]
EXP_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = EXP_DIR / "results"


def main() -> None:
    from data_loader import load_dataset
    from analysis import run_predictor_2a, validate_step1_vs_2a, run_predictor_2b

    RESULTS_DIR.mkdir(exist_ok=True)
    bundle = load_dataset()
    log.info("Dataset: %d pairs, %d cells, %d drugs",
             len(bundle.full_df), len(bundle.cell_order), len(bundle.drug_to_idx))

    result_2a = run_predictor_2a(bundle)
    step1_check = validate_step1_vs_2a(result_2a["mean_global_r"], RESULTS_DIR)
    result_2b = run_predictor_2b(RESULTS_DIR)

    result = {
        "predictor_2a": result_2a,
        "step1_vs_2a_validation": step1_check,
        "predictor_2b": result_2b if result_2b is not None else {
            "status": "pending",
            "note": "Run run_model_comparison.py first, then re-run this script.",
        },
    }

    out_path = RESULTS_DIR / "baseline_dissociation.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("Saved → %s", out_path)

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("Predictor 2A | global_r=%.4f  per_drug_r=0.000", result_2a["mean_global_r"])
    log.info("Step1↔2A    | theoretical=%.4f  empirical=%.4f  status=%s",
             step1_check.get("theoretical_ceiling", float("nan")),
             result_2a["mean_global_r"],
             step1_check.get("status", "N/A"))
    if result_2b:
        log.info("Predictor 2B | global_r=%.4f  per_drug_r=%.4f",
                 result_2b["mean_resid_global_r"], result_2b["mean_resid_per_drug_r"])
    else:
        log.info("Predictor 2B | pending (run run_model_comparison.py first)")


if __name__ == "__main__":
    main()
