"""Stage 2B: Modality ablation — medium capacity, best config from Stage 2A.

Variants (run one per sbatch job to cap peak memory):
  rna_only        RNA-seq only                (~11 GB pair matrix)
  rna_mut         RNA + mutations             (~17 GB)
  rna_mut_cnv     RNA + mutations + CNV       (~40 GB)
  rna_mut_cnv_met RNA + mutations + CNV + met (~41 GB)
  all_5_omics     All five modalities         (~43 GB)

Reads best medium config from results/stage2/capacity_sweep_best_configs.json if
available; falls back to pre-registered defaults (dropout=0.1, weight_decay=0.0).

Usage:
    # Single variant
    uv run python3 .../src/run_modality_ablation.py --variant rna_only
    # All variants sequentially (needs ~43 GB peak — runs each and frees before next)
    uv run python3 .../src/run_modality_ablation.py
    # Smoke test
    uv run python3 .../src/run_modality_ablation.py --variant rna_only --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).parents[4]
EXPERIMENT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from src.data.omics_utils import load_omics, z_score_normalize
from src.data.splits import cell_blind_split, drug_blind_split, mixed_set_split
from src.evaluation.metrics import evaluate_full
from src.train import predict_batched, train_mlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "experiments" / "01_metric_decomposition" / "03_baselines" / "results" / "stage2"
REPORT_DATA = REPO_ROOT / "experiments" / "01_metric_decomposition" / "03_baselines" / "report" / "data" / "metrics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096 if DEVICE == "cuda" else 2048
MAX_EPOCHS = 300
PATIENCE = 30
HIDDEN_DIMS = [512, 256, 64]  # medium capacity

MODALITY_VARIANTS: list[tuple[str, list[str]]] = [
    ("rna_only",        ["rna"]),
    ("rna_mut",         ["rna", "mutations"]),
    ("rna_mut_cnv",     ["rna", "mutations", "cnv"]),
    ("rna_mut_cnv_met", ["rna", "mutations", "cnv", "metabolomics"]),
    ("all_5_omics",     ["rna", "mutations", "cnv", "metabolomics", "rppa"]),
]

SPLIT_FNS = {
    "mixed_set": mixed_set_split,
    "cell_blind": cell_blind_split,
    "drug_blind": drug_blind_split,
}

# Fallback if capacity sweep has not run yet
_DEFAULT_CFG = {"dropout": 0.1, "weight_decay": 0.0}


def load_medium_config() -> dict:
    cfg_path = RESULTS_DIR / "capacity_sweep_best_configs.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text()).get("medium", _DEFAULT_CFG)
        logger.info("Loaded medium config from %s: %s", cfg_path, cfg)
    else:
        cfg = _DEFAULT_CFG
        logger.warning(
            "capacity_sweep_best_configs.json not found; using fallback %s. "
            "Run run_capacity_sweep.py first for the correct hyperparams.",
            cfg,
        )
    return cfg


def run_variant(
    variant_name: str,
    modalities: list[str],
    dr: pd.DataFrame,
    cell_lines: list[str],
    cfg: dict,
    smoke: bool = False,
) -> dict:
    logger.info("--- Variant: %s  modalities=%s ---", variant_name, modalities)
    cell_mat, cell_order = load_omics(modalities, cell_lines)
    # Map each pair row to its cell_mat row index. cell_mat is n_cells × n_features
    # (~280 MB even for all_5_omics), far cheaper than materializing the full
    # 151k-row pair matrix (~43 GB for rna_mut_cnv). Splits index directly into
    # cell_mat so pair_X is never allocated.
    c2r = {c: i for i, c in enumerate(cell_order)}
    pair_row_idx = np.array([c2r[c] for c in dr["depmap_id"]], dtype=np.intp)
    n_features = cell_mat.shape[1]
    logger.info("  features: %d", n_features)

    max_ep = 2 if smoke else MAX_EPOCHS
    splits_out: dict = {}
    Xtr = Xvl = Xte = None

    for split_name, split_fn in SPLIT_FNS.items():
        del Xtr, Xvl, Xte
        gc.collect()
        tr_i, vl_i, te_i = split_fn(dr, seed=42)
        Xtr, Xvl, Xte = z_score_normalize(
            cell_mat[pair_row_idx[tr_i]],
            cell_mat[pair_row_idx[vl_i]],
            cell_mat[pair_row_idx[te_i]],
        )
        ytr = dr.iloc[tr_i]["ln_ic50"].to_numpy(dtype=np.float32)
        yvl = dr.iloc[vl_i]["ln_ic50"].to_numpy(dtype=np.float32)
        val_drugs = dr.iloc[vl_i]["drug_name"].to_numpy()

        model, curve, _, best_epoch = train_mlp(
            Xtr, ytr, Xvl, yvl, val_drugs,
            HIDDEN_DIMS, float(cfg["dropout"]), float(cfg["weight_decay"]), DEVICE,
            max_epochs=max_ep, patience=PATIENCE, batch_size=BATCH_SIZE,
        )
        preds = predict_batched(model, Xte, DEVICE, BATCH_SIZE)
        y_te = dr.iloc[te_i]["ln_ic50"].to_numpy(dtype=np.float32)
        m = dict(evaluate_full(
            y_te, preds,
            dr.iloc[te_i]["drug_name"].to_numpy(),
            dr.iloc[te_i]["depmap_id"].to_numpy(),
        ))
        m["val_per_drug_r_curve"] = [round(v, 5) if not np.isnan(v) else None for v in curve]
        m["best_epoch"] = best_epoch
        m["converged"] = best_epoch < MAX_EPOCHS - PATIENCE
        splits_out[split_name] = m
        logger.info(
            "  [%s]  per_drug_r=%.4f  global_r=%.4f  best_epoch=%d  converged=%s",
            split_name, m["per_drug_r"], m["global_r"], best_epoch, m["converged"],
        )

    return {"modalities": modalities, "n_features": int(n_features), "splits": splits_out}


def main(variant: str | None = None, smoke: bool = False) -> None:
    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    cell_lines = overlap["depmap_id"].tolist()

    dr = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    dr = pd.DataFrame(dr[dr["depmap_id"].isin(cell_lines)].reset_index(drop=True))

    if smoke:
        dr = dr.sample(n=min(2000, len(dr)), random_state=0).reset_index(drop=True)
        logger.info("[SMOKE] Subsample: %d pairs", len(dr))

    cfg = load_medium_config()
    logger.info("Device: %s  Pairs: %d", DEVICE, len(dr))

    variants_to_run = MODALITY_VARIANTS
    if variant is not None:
        variants_to_run = [(n, m) for n, m in MODALITY_VARIANTS if n == variant]
        if not variants_to_run:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {[n for n, _ in MODALITY_VARIANTS]}")

    t0 = time.perf_counter()
    all_results: dict = {}

    for variant_name, modalities in variants_to_run:
        all_results[variant_name] = run_variant(variant_name, modalities, dr, cell_lines, cfg, smoke=smoke)
        # Explicit loop boundary — pair_X is re-created each iteration and freed here.

    logger.info("Total runtime: %.1f min", (time.perf_counter() - t0) / 60)

    if smoke:
        logger.info("[SMOKE] Pipeline OK — not writing results.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Merge with any previously saved variants (allows running variants separately)
    out = RESULTS_DIR / "modality_ablation_results.json"
    existing_ablation = json.loads(out.read_text()) if out.exists() else {}
    existing_ablation.update(all_results)
    out.write_text(json.dumps(existing_ablation, indent=2))
    logger.info("Results: %s", out)

    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT_DATA.read_text()) if REPORT_DATA.exists() else {}
    report.setdefault("modality_ablation", {}).update(all_results)
    REPORT_DATA.write_text(json.dumps(report, indent=2))
    logger.info("Report data: %s", REPORT_DATA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=[n for n, _ in MODALITY_VARIANTS],
        default=None,
        help="Run a single modality variant. Omit to run all five sequentially.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(variant=args.variant, smoke=args.smoke)
