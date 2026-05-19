"""Train and evaluate all three baselines under three split protocols.

Usage:
    python experiments/01_metric_decomposition/01_baselines/jobs/run_baselines.py [--resume RUN_DIR]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import MultiOmicsDataset
from src.data.splits import cell_blind_split, drug_blind_split, mixed_set_split
from src.evaluation.metrics import evaluate
from src.models.mlp import ConcatenationBaseline, LateFusionBaseline, RNAOnlyBaseline
from src.training.trainer import DEVICE, predict, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "results"
REPORT_DATA = REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "report" / "data" / "metrics.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SPLIT_FNS = {
    "mixed_set": mixed_set_split,
    "cell_blind": cell_blind_split,
    "drug_blind": drug_blind_split,
}

N_EPOCHS = 50
LR = 1e-3


def estimate_runtime(dataset, n_epochs=N_EPOCHS) -> float:
    """Return estimated total seconds based on a 3-batch timing probe."""
    from src.models.mlp import ConcatenationBaseline
    from src.training.trainer import _Prefetcher, _build_concat, DEFAULT_BATCH_SIZE, _sync
    import torch, torch.nn as nn

    bs = DEFAULT_BATCH_SIZE.get(DEVICE, 2048)
    concat_np = _build_concat(dataset)
    dummy_idx = np.arange(min(bs * 10, len(dataset)))
    model = ConcatenationBaseline(dataset.feature_dims, dataset.omics_to_use).to(DEVICE)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())
    prefetcher = _Prefetcher(concat_np, dataset._cell_rows, dataset._targets,
                             dummy_idx, bs, DEVICE)
    t0 = time.perf_counter()
    for _ in range(5):
        x, y = next(prefetcher)
        opt.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        opt.step()
    _sync(DEVICE)
    prefetcher.stop()
    secs_per_step = (time.perf_counter() - t0) / 5
    steps_per_epoch = len(dataset) // bs
    secs_per_epoch = secs_per_step * steps_per_epoch
    # 3 models × 3 splits × n_epochs
    total = 9 * n_epochs * secs_per_epoch
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing run dir to resume from")
    args = parser.parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        logger.info("Resuming run: %s", run_dir)
        results_path = run_dir / "results.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else {}
    else:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        logger.info("Run dir: %s", run_dir)

    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    cell_lines = overlap["depmap_id"].tolist()
    logger.info("Cell lines with all 5 omics: %d", len(cell_lines))

    dataset = MultiOmicsDataset(cell_lines=cell_lines, modality_dropout_p=0.0)
    logger.info(
        "Dataset: %d pairs  %d drugs  %d cell lines  device=%s",
        len(dataset), dataset.n_drugs, len(cell_lines), DEVICE,
    )

    # Estimate runtime before starting
    est_secs = estimate_runtime(dataset, N_EPOCHS)
    logger.info(
        "Estimated total runtime: %.0f min (%.0f h)  [%d epochs × 9 runs]",
        est_secs / 60, est_secs / 3600, N_EPOCHS,
    )

    run_t0 = time.perf_counter()

    for split_name, split_fn in SPLIT_FNS.items():
        logger.info("\n=== Split: %s ===", split_name)
        train_idx, val_idx, test_idx = split_fn(dataset.pairs)
        logger.info("  train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

        if split_name not in results:
            results[split_name] = {}

        test_targets = dataset.get_targets(test_idx)

        model_configs = [
            ("rna_only",      RNAOnlyBaseline(dataset.feature_dims, dataset.omics_to_use)),
            ("concatenation", ConcatenationBaseline(dataset.feature_dims, dataset.omics_to_use)),
            ("late_fusion",   LateFusionBaseline(dataset.feature_dims, dataset.omics_to_use)),
        ]

        for model_name, model in model_configs:
            if model_name in results.get(split_name, {}):
                logger.info("  [%s] already done, skipping", model_name)
                continue

            logger.info("  [%s]", model_name)
            ckpt_dir = run_dir / "checkpoints"
            resume_path = ckpt_dir / f"{split_name}_{model_name}_final.pt"
            resume = resume_path if resume_path.exists() else None

            train(
                model, dataset, train_idx, val_idx,
                n_epochs=N_EPOCHS, lr=LR,
                run_dir=run_dir,
                checkpoint_every=10,
                resume_from=resume,
                model_name=f"{split_name}_{model_name}",
            )

            test_preds = predict(model, dataset, test_idx)
            results[split_name][model_name] = evaluate(test_targets, test_preds)

            # Save results after each model completes (crash-safe)
            (run_dir / "results.json").write_text(json.dumps(results, indent=2))
            logger.info(
                "  %-15s  r=%.4f  rmse=%.4f  n=%d",
                model_name,
                results[split_name][model_name]["pearson_r"],
                results[split_name][model_name]["rmse"],
                results[split_name][model_name]["n"],
            )

    total_mins = (time.perf_counter() - run_t0) / 60
    logger.info("\nTotal runtime: %.1f min", total_mins)

    # Summary table
    print("\n\n=== BASELINE RESULTS SUMMARY ===\n")
    header = f"{'Model':<22} {'mixed_set r':>12} {'cell_blind r':>13} {'drug_blind r':>13}"
    print(header)
    print("-" * len(header))
    for model_name in ["rna_only", "concatenation", "late_fusion"]:
        row = f"{model_name:<22}"
        for split in ["mixed_set", "cell_blind", "drug_blind"]:
            r = results.get(split, {}).get(model_name, {}).get("pearson_r", float("nan"))
            row += f"  {r:>11.4f}"
        print(row)
    print(f"\n{'PASO (literature)':<22}  {'0.9425':>11} {'0.8869':>13} {'0.7448':>13}")
    print(f"{'DeepCDR (literature)':<22}  {'0.923':>11} {'0.889':>13} {'—':>13}")
    print(f"\nResults: {run_dir / 'results.json'}")

    # Write to report data so render_reports.py picks it up automatically
    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DATA.write_text(json.dumps(results, indent=2))
    logger.info("Report data updated: %s", REPORT_DATA)


if __name__ == "__main__":
    main()
