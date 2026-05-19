"""Dissociation analysis functions for Step 2 (baseline_dissociation job).

Predictor 2A: per-drug-mean predictor (no model).
Predictor 2B: drug-mean-removed model predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def run_predictor_2a(bundle, n_folds: int = 5) -> dict:
    """Per-drug-mean predictor on mixed_set. Policy: undefined per-drug r = 0.0."""
    from scipy.stats import pearsonr
    from data_loader import make_random_folds

    log.info("=== Predictor 2A: per-drug-mean on mixed_set ===")
    folds = make_random_folds(bundle.full_df, "mixed_set", n_folds=n_folds)
    fold_results = []

    for fold_i, (train_idx, _val_idx, test_idx) in enumerate(folds):
        train_drugs = bundle.drug_names[train_idx]
        train_targets = bundle.targets[train_idx]
        global_mean = float(train_targets.mean())

        drug_means: dict[str, float] = {}
        for d in np.unique(train_drugs):
            drug_means[d] = float(train_targets[train_drugs == d].mean())

        test_drugs = bundle.drug_names[test_idx]
        test_targets = bundle.targets[test_idx]
        preds = np.array([drug_means.get(d, global_mean) for d in test_drugs], dtype=np.float32)

        global_r = float(pearsonr(preds, test_targets)[0])
        fold_results.append({
            "fold": fold_i,
            "global_r": round(global_r, 5),
            "per_drug_r_mean": 0.0,
            "gap": round(0.0 - global_r, 5),
            "n_test_drugs": int(len(np.unique(test_drugs))),
            "note": "per_drug_r undefined (constant prediction) → imputed 0.0 by policy",
        })
        log.info("  Fold %d | global_r=%.4f  per_drug_r=0.000 (undefined→0)", fold_i, global_r)

    global_rs = [r["global_r"] for r in fold_results]
    mean_global_r = float(np.mean(global_rs))
    log.info("2A mean global_r=%.4f ± %.4f", mean_global_r, float(np.std(global_rs, ddof=1)))
    return {
        "description": "per-drug-mean predictor on mixed_set (no model)",
        "per_drug_r_policy": "undefined constant prediction → imputed 0.0",
        "folds": fold_results,
        "mean_global_r": round(mean_global_r, 5),
        "std_global_r": round(float(np.std(global_rs, ddof=1)), 5),
        "mean_per_drug_r": 0.0,
        "mean_gap": round(0.0 - mean_global_r, 5),
    }


def validate_step1_vs_2a(mean_global_r: float, results_dir: Path) -> dict:
    """Check if 2A empirical global r matches Step 1 theoretical ceiling (≤ ±0.02)."""
    step1_path = results_dir / "variance_decomposition.json"
    if not step1_path.exists():
        log.warning("Step 1 results not found — skipping validation")
        return {"status": "skipped", "reason": "variance_decomposition.json not found"}

    with step1_path.open() as f:
        step1 = json.load(f)

    theoretical = float(step1["global_r_ceiling_from_between"])
    diff = abs(mean_global_r - theoretical)
    status = "PASS" if diff <= 0.02 else "FAIL"
    log.info("Step1↔2A: theoretical=%.4f empirical=%.4f diff=%.4f → %s",
             theoretical, mean_global_r, diff, status)
    return {
        "status": status,
        "theoretical_ceiling": theoretical,
        "empirical_global_r_2a": round(mean_global_r, 5),
        "abs_diff": round(diff, 5),
        "threshold": 0.02,
    }


def run_predictor_2b(results_dir: Path) -> Optional[dict]:
    """Drug-mean-removed TransformerEncoder predictions saved by run_model_comparison."""
    from scipy.stats import pearsonr
    from src.evaluation.per_drug import per_drug_r as compute_per_drug_r

    mc_dirs = sorted(results_dir.glob("model_comparison_*"))
    if not mc_dirs:
        log.warning("No model_comparison_* dir found. Run run_model_comparison.py first.")
        return None

    mc_dir = mc_dirs[-1]
    pred_files = sorted(mc_dir.glob("predictions_TransformerEncoder_fold*.npz"))
    if not pred_files:
        log.warning("No TransformerEncoder .npz files in %s", mc_dir)
        return None

    log.info("=== Predictor 2B: drug-mean-removed TransformerEncoder (drug_blind) ===")
    fold_results = []

    for pf in pred_files:
        fold_i = int(pf.stem.split("fold")[1])
        data = np.load(pf, allow_pickle=True)
        preds = data["preds"]
        targets = data["targets"]
        drug_names = data["drug_names"]

        orig_global_r = float(pearsonr(preds, targets)[0])
        orig_rs = compute_per_drug_r(preds, targets, drug_names, min_cells=5)
        orig_per_drug_r = float(np.mean(list(orig_rs.values()))) if orig_rs else float("nan")

        # Subtract per-drug mean of test predictions: ŷ_resid(c,d) = ŷ(c,d) - mean_{c'}ŷ(c',d)
        preds_resid = preds.copy()
        for d in np.unique(drug_names):
            mask = drug_names == d
            preds_resid[mask] -= preds[mask].mean()

        resid_global_r = float(pearsonr(preds_resid, targets)[0])
        resid_rs = compute_per_drug_r(preds_resid, targets, drug_names, min_cells=5)
        resid_per_drug_r = float(np.mean(list(resid_rs.values()))) if resid_rs else float("nan")

        fold_results.append({
            "fold": fold_i,
            "original": {"global_r": round(orig_global_r, 5), "per_drug_r": round(orig_per_drug_r, 5)},
            "drug_mean_removed": {"global_r": round(resid_global_r, 5), "per_drug_r": round(resid_per_drug_r, 5)},
        })
        log.info("  Fold %d | orig global=%.4f pd=%.4f | resid global=%.4f pd=%.4f",
                 fold_i, orig_global_r, orig_per_drug_r, resid_global_r, resid_per_drug_r)

    resid_globals = [r["drug_mean_removed"]["global_r"] for r in fold_results]
    resid_pds = [r["drug_mean_removed"]["per_drug_r"] for r in fold_results]
    log.info("2B | resid global_r=%.4f ± %.4f  per_drug_r=%.4f ± %.4f",
             np.mean(resid_globals), np.std(resid_globals, ddof=1),
             np.mean(resid_pds), np.std(resid_pds, ddof=1))

    orig_globals = [r["original"]["global_r"] for r in fold_results]
    return {
        "description": "drug-mean-removed TransformerEncoder predictions on drug_blind",
        "operation": "ŷ_resid(c,d) = ŷ(c,d) − mean_{c' in test_d}(ŷ(c',d))",
        "source_dir": str(mc_dir),
        "folds": fold_results,
        "mean_original_global_r": round(float(np.mean(orig_globals)), 5),
        "mean_resid_global_r": round(float(np.mean(resid_globals)), 5),
        "mean_resid_per_drug_r": round(float(np.mean(resid_pds)), 5),
        "interpretation": (
            "Removing per-drug mean from predictions collapses global_r toward 0 "
            "while preserving per_drug_r — confirming the two metrics measure "
            "independent signals."
        ),
    }
