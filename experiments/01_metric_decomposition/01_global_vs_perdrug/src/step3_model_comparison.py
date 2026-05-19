"""Step 3: Multi-model consistency — entry point.

Logic lives in src/runner.py.
Features: PCA(RNA,550) + PCA(mut,200) + MorganFP(2048) for Ridge/MLP; raw for TransformerEncoder.
Split: PASO drug_blind, 5 folds.

Runtime: ~4–6 h on NVIDIA GB10.
Output:  results/model_comparison_<timestamp>/
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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
    parser.add_argument("--models", nargs="+",
                        default=["Ridge", "MLP-S", "MLP-M", "MLP-L", "TransformerEncoder"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or _detect_device()
    log.info("Device: %s", device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"model_comparison_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "logs").mkdir(exist_ok=True)

    fh = logging.FileHandler(EXP_DIR / "logs" / "model_comparison.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    from data_loader import load_dataset, load_dataset_pca, make_paso_drug_blind_folds
    from runner import (run_ridge, run_mlp, run_transformer_encoder, decision_check_models)

    log.info("Loading dataset + PCA compression...")
    bundle, pca_omics = load_dataset_pca(rna_dim=550, mut_dim=200)
    folds = make_paso_drug_blind_folds(bundle.full_df, bundle.key_to_idx, bundle.name_to_depmap)
    log.info("Dataset: %d pairs, %d cells, %d drugs | PCA shape: %s",
             len(bundle.full_df), len(bundle.cell_order), len(bundle.drug_to_idx), pca_omics.shape)

    X_all = np.concatenate([pca_omics[bundle.cell_rows], bundle.fp_matrix[bundle.drug_idxs]], axis=1)
    log.info("Pair feature matrix: %s  (%.1f MB)", X_all.shape, X_all.nbytes / 1e6)

    results: dict[str, dict] = {}
    for model_name in args.models:
        if model_name == "Ridge":
            results["Ridge"] = run_ridge(X_all, bundle.targets, bundle.drug_names, folds, run_dir)
        elif model_name.startswith("MLP-"):
            size = model_name.split("-")[1]
            results[model_name] = run_mlp(X_all, bundle.targets, bundle.drug_names,
                                          folds, run_dir, size, device)
        elif model_name == "TransformerEncoder":
            results["TransformerEncoder"] = run_transformer_encoder(bundle, folds, run_dir, device)

        # Checkpoint after each model
        with (run_dir / "results_partial.json").open("w") as f:
            json.dump(results, f, indent=2)

    dc = decision_check_models(results)
    output = {
        "timestamp": timestamp, "device": device,
        "feature_set": "PCA(RNA,550)+PCA(mut,200)+MorganFP(2048) [Ridge/MLP]; raw [TransformerEncoder]",
        "split": "PASO drug_blind", "n_folds": 5,
        "models": results, "decision_criterion": dc,
    }

    with (run_dir / "results.json").open("w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved → %s", run_dir / "results.json")

    log.info("=" * 70)
    log.info("%-16s  global_r   per_drug_r  gap      95%%CI            p-value", "Model")
    log.info("-" * 70)
    for name, r in results.items():
        log.info("%-16s  %.4f     %.4f      %.4f   [%.4f,%.4f]  %.4f",
                 name, r["mean_global_r"], r["mean_per_drug_r"], r["mean_gap"],
                 r["gap_95ci"][0], r["gap_95ci"][1], r["gap_ttest_p"])
    log.info("Decision (gap > 0.10 for all): %s", "PASS" if dc["all_pass"] else "FAIL")


if __name__ == "__main__":
    main()
