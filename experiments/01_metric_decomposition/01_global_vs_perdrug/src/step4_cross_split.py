"""Step 4: Cross-split consistency — entry point.

Logic lives in src/runner.py.
TransformerEncoder trained on 3 splits × 5 folds.

Runtime: ~3 h on NVIDIA GB10.
Output:  results/cross_split_<timestamp>/  +  results/cross_split_consistency.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parents[4]
EXP_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = EXP_DIR / "results"


def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+",
                        default=["drug_blind", "mixed_set", "cell_blind"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or _detect_device()
    log.info("Device: %s", device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"cross_split_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "logs").mkdir(exist_ok=True)

    fh = logging.FileHandler(EXP_DIR / "logs" / "cross_split.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    from data_loader import load_dataset, make_paso_drug_blind_folds, make_random_folds
    from runner import run_split, decision_check_splits

    log.info("Loading dataset...")
    bundle = load_dataset()
    log.info("Dataset: %d pairs, %d cells, %d drugs",
             len(bundle.full_df), len(bundle.cell_order), len(bundle.drug_to_idx))

    split_folds: dict[str, list] = {}
    for split_name in args.splits:
        if split_name == "drug_blind":
            split_folds[split_name] = make_paso_drug_blind_folds(
                bundle.full_df, bundle.key_to_idx, bundle.name_to_depmap)
        else:
            split_folds[split_name] = make_random_folds(bundle.full_df, split_name)

    results: dict[str, dict] = {}
    for split_name in args.splits:
        results[split_name] = run_split(split_name, split_folds[split_name],
                                        bundle, run_dir, device)
        with (run_dir / "results_partial.json").open("w") as f:
            json.dump(results, f, indent=2)

    dc = decision_check_splits(results)
    output = {
        "timestamp": timestamp, "device": device,
        "model": "TransformerEncoder", "n_folds": 5,
        "splits": args.splits, "results": results, "decision_criterion": dc,
    }

    with (run_dir / "results.json").open("w") as f:
        json.dump(output, f, indent=2)

    summary_path = RESULTS_DIR / "cross_split_consistency.json"
    with summary_path.open("w") as f:
        json.dump(output, f, indent=2)

    log.info("Saved → %s", run_dir / "results.json")
    log.info("Saved → %s", summary_path)

    log.info("=" * 65)
    log.info("%-12s  global_r   per_drug_r  gap      95%%CI            p-value", "Split")
    log.info("-" * 65)
    for split_name, r in results.items():
        log.info("%-12s  %.4f     %.4f      %.4f   [%.4f,%.4f]  %.4f",
                 split_name, r["mean_global_r"], r["mean_per_drug_r"], r["mean_gap"],
                 r["gap_95ci"][0], r["gap_95ci"][1], r["gap_ttest_p"])
    log.info("Decision (gap > 0.05 on all splits): %s", "PASS" if dc["all_pass"] else "FAIL")


if __name__ == "__main__":
    main()
