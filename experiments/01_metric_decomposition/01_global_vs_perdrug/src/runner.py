"""Model and split runner functions for Steps 3 and 4.

run_ridge, run_mlp, run_transformer_encoder → Step 3 (model_comparison job)
run_split                      → Step 4 (cross_split job)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------------------
# Step 3: model runners
# ---------------------------------------------------------------------------

def run_ridge(
    X_all: np.ndarray,
    y_all: np.ndarray,
    drug_names_all: np.ndarray,
    folds: list,
    run_dir: Path,
) -> dict:
    from scipy.stats import pearsonr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from data_loader import compute_metrics, bootstrap_ci, paired_ttest

    log.info("=== Ridge ===")
    fold_results = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(folds):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_all[train_idx])
        X_va = sc.transform(X_all[val_idx])
        X_te = sc.transform(X_all[test_idx])

        best_alpha, best_val_r = ALPHA_GRID[0], -np.inf
        for alpha in ALPHA_GRID:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr, y_all[train_idx])
            val_r = float(pearsonr(ridge.predict(X_va), y_all[val_idx])[0])
            if val_r > best_val_r:
                best_val_r, best_alpha = val_r, alpha

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_tr, y_all[train_idx])
        test_preds = ridge.predict(X_te).astype(np.float32)

        m = compute_metrics(test_preds, y_all[test_idx], drug_names_all[test_idx])
        m["fold"] = fold_i
        m["best_alpha"] = best_alpha
        fold_results.append(m)

        np.savez_compressed(
            run_dir / f"predictions_Ridge_fold{fold_i}.npz",
            preds=test_preds, targets=y_all[test_idx], drug_names=drug_names_all[test_idx],
        )
        log.info("  Ridge fold %d | alpha=%.2g  global_r=%.4f  per_drug_r=%.4f  gap=%.4f",
                 fold_i, best_alpha, m["global_r"], m["per_drug_r_mean"], m["gap"])

    gaps = [r["gap"] for r in fold_results]
    ci_lo, ci_hi = bootstrap_ci(gaps)
    t_stat, p_val = paired_ttest(gaps)
    log.info("Ridge summary | mean_gap=%.4f  CI=[%.4f,%.4f]  t=%.3f  p=%.4f",
             float(np.mean(gaps)), ci_lo, ci_hi, t_stat, p_val)
    return _summarise("Ridge", fold_results, gaps, ci_lo, ci_hi, t_stat, p_val)


def run_mlp(
    X_all: np.ndarray,
    y_all: np.ndarray,
    drug_names_all: np.ndarray,
    folds: list,
    run_dir: Path,
    size: str,
    device: str,
) -> dict:
    from cell_mlp import train_mlp_fold
    from data_loader import compute_metrics, bootstrap_ci, paired_ttest

    log.info("=== MLP-%s ===", size)
    fold_results = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(folds):
        test_preds, test_targets, test_drugs = train_mlp_fold(
            X_all, y_all, drug_names_all,
            train_idx, val_idx, test_idx,
            size=size, device=device, fold_label=f"fold{fold_i}",
        )
        m = compute_metrics(test_preds, test_targets, test_drugs)
        m["fold"] = fold_i
        fold_results.append(m)

        np.savez_compressed(
            run_dir / f"predictions_MLP{size}_fold{fold_i}.npz",
            preds=test_preds, targets=test_targets, drug_names=test_drugs,
        )
        log.info("  MLP-%s fold %d | global_r=%.4f  per_drug_r=%.4f  gap=%.4f",
                 size, fold_i, m["global_r"], m["per_drug_r_mean"], m["gap"])

    gaps = [r["gap"] for r in fold_results]
    ci_lo, ci_hi = bootstrap_ci(gaps)
    t_stat, p_val = paired_ttest(gaps)
    log.info("MLP-%s summary | mean_gap=%.4f  CI=[%.4f,%.4f]  t=%.3f  p=%.4f",
             size, float(np.mean(gaps)), ci_lo, ci_hi, t_stat, p_val)
    return _summarise(f"MLP-{size}", fold_results, gaps, ci_lo, ci_hi, t_stat, p_val)


def run_transformer_encoder(
    bundle,
    folds: list,
    run_dir: Path,
    device: str,
) -> dict:
    from omnicancer_trainer import train_transformer_fold
    from data_loader import compute_metrics, bootstrap_ci, paired_ttest

    log.info("=== TransformerEncoder ===")
    fold_results = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(folds):
        test_preds, test_targets, test_drugs, _ = train_transformer_fold(
            concat_np=bundle.concat_np, cell_rows=bundle.cell_rows,
            drug_idxs=bundle.drug_idxs, fp_matrix=bundle.fp_matrix,
            targets=bundle.targets, drug_names_all=bundle.drug_names,
            cell_ids_all=bundle.cell_ids, feature_dims=bundle.feature_dims,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            device=device, fold_label=f"fold{fold_i}",
        )
        m = compute_metrics(test_preds, test_targets, test_drugs)
        m["fold"] = fold_i
        fold_results.append(m)

        np.savez_compressed(
            run_dir / f"predictions_TransformerEncoder_fold{fold_i}.npz",
            preds=test_preds, targets=test_targets, drug_names=test_drugs,
        )
        log.info("  TransformerEncoder fold %d | global_r=%.4f  per_drug_r=%.4f  gap=%.4f",
                 fold_i, m["global_r"], m["per_drug_r_mean"], m["gap"])

    gaps = [r["gap"] for r in fold_results]
    ci_lo, ci_hi = bootstrap_ci(gaps)
    t_stat, p_val = paired_ttest(gaps)
    log.info("TransformerEncoder summary | mean_gap=%.4f  CI=[%.4f,%.4f]  t=%.3f  p=%.4f",
             float(np.mean(gaps)), ci_lo, ci_hi, t_stat, p_val)
    return _summarise("TransformerEncoder", fold_results, gaps, ci_lo, ci_hi, t_stat, p_val)


# ---------------------------------------------------------------------------
# Step 4: split runner
# ---------------------------------------------------------------------------

def run_split(
    split_name: str,
    folds: list,
    bundle,
    run_dir: Path,
    device: str,
) -> dict:
    from omnicancer_trainer import train_transformer_fold
    from data_loader import compute_metrics, bootstrap_ci, paired_ttest

    log.info("=== Split: %s ===", split_name)
    fold_results = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(folds):
        test_preds, test_targets, test_drugs, test_cells = train_transformer_fold(
            concat_np=bundle.concat_np, cell_rows=bundle.cell_rows,
            drug_idxs=bundle.drug_idxs, fp_matrix=bundle.fp_matrix,
            targets=bundle.targets, drug_names_all=bundle.drug_names,
            cell_ids_all=bundle.cell_ids, feature_dims=bundle.feature_dims,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            device=device, fold_label=f"{split_name}_fold{fold_i}",
        )
        m = compute_metrics(test_preds, test_targets, test_drugs)
        m["fold"] = fold_i
        fold_results.append(m)

        np.savez_compressed(
            run_dir / f"predictions_{split_name}_fold{fold_i}.npz",
            preds=test_preds, targets=test_targets,
            drug_names=test_drugs, cell_ids=test_cells,
        )
        log.info("  %s fold %d | global_r=%.4f  per_drug_r=%.4f  gap=%.4f",
                 split_name, fold_i, m["global_r"], m["per_drug_r_mean"], m["gap"])

    gaps = [r["gap"] for r in fold_results]
    ci_lo, ci_hi = bootstrap_ci(gaps)
    t_stat, p_val = paired_ttest(gaps)
    log.info("%s summary | mean_gap=%.4f  CI=[%.4f,%.4f]  t=%.3f  p=%.4f",
             split_name, float(np.mean(gaps)), ci_lo, ci_hi, t_stat, p_val)

    global_rs = [r["global_r"] for r in fold_results]
    per_drug_rs = [r["per_drug_r_mean"] for r in fold_results]
    return {
        "split": split_name,
        "folds": fold_results,
        "mean_global_r": round(float(np.mean(global_rs)), 5),
        "std_global_r": round(float(np.std(global_rs, ddof=1)), 5),
        "mean_per_drug_r": round(float(np.mean(per_drug_rs)), 5),
        "std_per_drug_r": round(float(np.std(per_drug_rs, ddof=1)), 5),
        "mean_gap": round(float(np.mean(gaps)), 5),
        "gap_95ci": [round(ci_lo, 5), round(ci_hi, 5)],
        "gap_ttest_t": round(t_stat, 4),
        "gap_ttest_p": round(p_val, 6),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _summarise(
    name: str,
    fold_results: list[dict],
    gaps: list[float],
    ci_lo: float,
    ci_hi: float,
    t_stat: float,
    p_val: float,
) -> dict:
    global_rs = [r["global_r"] for r in fold_results]
    per_drug_rs = [r["per_drug_r_mean"] for r in fold_results]
    return {
        "model": name,
        "folds": fold_results,
        "mean_global_r": round(float(np.mean(global_rs)), 5),
        "std_global_r": round(float(np.std(global_rs, ddof=1)), 5),
        "mean_per_drug_r": round(float(np.mean(per_drug_rs)), 5),
        "std_per_drug_r": round(float(np.std(per_drug_rs, ddof=1)), 5),
        "mean_gap": round(float(np.mean(gaps)), 5),
        "gap_95ci": [round(ci_lo, 5), round(ci_hi, 5)],
        "gap_ttest_t": round(t_stat, 4),
        "gap_ttest_p": round(p_val, 6),
    }


def decision_check_models(results: dict) -> dict:
    """Pre-registered: gap > 0.10 for all model classes → metric structure confirmed."""
    checks, all_pass = {}, True
    for name, r in results.items():
        gap = r["mean_gap"]
        passed = gap > 0.10
        checks[name] = {"mean_gap": gap, "threshold": 0.10, "pass": passed}
        if not passed:
            all_pass = False
    return {"all_pass": all_pass, "per_model": checks}


def decision_check_splits(results: dict) -> dict:
    """Pre-registered: gap > 0.05 on all splits → not a split-specific artifact."""
    checks, all_pass = {}, True
    for split, r in results.items():
        gap = r["mean_gap"]
        passed = gap > 0.05
        checks[split] = {"mean_gap": gap, "threshold": 0.05, "pass": passed}
        if not passed:
            all_pass = False
    return {"all_pass": all_pass, "per_split": checks}
