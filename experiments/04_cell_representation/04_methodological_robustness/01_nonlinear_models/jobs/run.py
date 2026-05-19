"""01_nonlinear_models: XGBoost and MLP vs Ridge on PASO drug-blind CV.

Tests whether the drug-blind per-drug r=0.631 ceiling is Ridge-specific (linear model)
or holds across model classes. Same RNA PCA(550) + mut PCA(200) features throughout.

Usage:
  python run.py                        # all conditions → results.json
  python run.py --condition ridge      # single condition → results_ridge.json
  python run.py --smoke                # 1 fold only
  python run.py --smoke --condition xgboost

Output: report/data/results.json  (or results_<condition>.json)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.evaluation.per_drug import mean_per_drug_r  # noqa: E402
from src.utils.paso_folds import load_cell_line_index, load_paso_pairs  # noqa: E402
from src.utils.ridge import compress_cell, safe_fit_scaler  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"
EXP_DIR = Path(__file__).parents[1]

K_FOLDS = 10
RNA_DIM = 550
MUT_DIM = 200
MIN_CELLS = 5

ALL_CONDITIONS = ["ridge", "xgboost", "mlp"]


def build_model(name: str) -> Ridge | xgb.XGBRegressor | MLPRegressor:
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
    msg = f"Unknown model: {name}"
    raise ValueError(msg)


def run_fold(
    fold_i: int,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    available_cells: set[str],
    name_to_depmap: dict[str, str],
    active_conditions: list[str],
) -> dict:
    train_df, test_df = load_paso_pairs(
        PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
    )
    if len(train_df) == 0 or len(test_df) == 0:
        logger.warning("fold %d: empty split — skip", fold_i)
        return {}

    all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
    train_cells = sorted(train_df["depmap_id"].unique())
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    train_rows = np.array([cell_to_row[c] for c in train_cells], dtype=np.int32)

    rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_rows, rna_dim=RNA_DIM, mut_dim=MUT_DIM)
    cell_feat = np.concatenate([rna_pca, mut_pca], axis=1)

    tr_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
    te_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
    y_train = train_df["ln_ic50"].values.astype(np.float32)
    y_test = test_df["ln_ic50"].values.astype(np.float32)
    drug_names = test_df["drug_name"].values

    sc = safe_fit_scaler(cell_feat[tr_idx])
    X_train = sc.transform(cell_feat[tr_idx])
    X_test = sc.transform(cell_feat[te_idx])

    fold_result: dict = {"fold": fold_i}
    preds_by_cond: dict[str, np.ndarray] = {}

    for cond in active_conditions:
        model = build_model(cond)
        model.fit(X_train, y_train)
        preds = model.predict(X_test).astype(np.float32)
        preds_by_cond[cond] = preds
        r = float(mean_per_drug_r(preds, y_test, drug_names, min_cells=MIN_CELLS))
        logger.info("  fold %d [%-8s]: per_drug_r=%.4f", fold_i, cond, r)
        fold_result[cond] = {"per_drug_r": r}

    # Fold-level prediction correlation (only when all conditions present)
    if "ridge" in preds_by_cond:
        for cond in ["xgboost", "mlp"]:
            if cond in preds_by_cond:
                corr = float(pearsonr(preds_by_cond["ridge"], preds_by_cond[cond]).statistic)
                fold_result[f"pred_corr_ridge_{cond}"] = corr
                logger.info("  fold %d pred_corr(ridge, %s)=%.4f", fold_i, cond, corr)

    return fold_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 fold only")
    parser.add_argument("--condition", type=str, choices=ALL_CONDITIONS, default=None,
                        help="Run a single condition (writes results_<condition>.json)")
    args = parser.parse_args()
    k_folds = 1 if args.smoke else K_FOLDS
    active_conditions = [args.condition] if args.condition else ALL_CONDITIONS

    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(logs_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("01_nonlinear_models: XGBoost + MLP vs Ridge, PASO drug-blind per-drug r%s%s",
                f" [cond={args.condition}]" if args.condition else "",
                " [SMOKE]" if args.smoke else "")

    if args.condition:
        results_path = report_dir / f"results_{args.condition}.json"
    else:
        results_path = report_dir / "results.json"

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    available_cells = set(rna.index) & set(mutations.index)
    logger.info("available cells: %d", len(available_cells))

    name_to_depmap = load_cell_line_index(DATA_DIR)

    all_folds: list[dict] = []

    for fold_i in range(k_folds):
        logger.info("=== Fold %d/%d ===", fold_i + 1, k_folds)
        fold_res = run_fold(fold_i, rna, mutations, available_cells, name_to_depmap,
                            active_conditions)
        if fold_res:
            all_folds.append(fold_res)
            results_path.write_text(json.dumps({"fold_results": all_folds}, indent=2))

    # Summary
    logger.info("=" * 70)
    cond_vals: dict[str, list[float]] = {c: [] for c in active_conditions}
    corr_vals: dict[str, list[float]] = {c: [] for c in ["xgboost", "mlp"]
                                          if c in active_conditions and "ridge" in active_conditions}

    for r in all_folds:
        for cond in active_conditions:
            if cond in r:
                cond_vals[cond].append(r[cond]["per_drug_r"])
        for cond in list(corr_vals.keys()):
            key = f"pred_corr_ridge_{cond}"
            if key in r:
                corr_vals[cond].append(r[key])

    ridge_mean = float(np.mean(cond_vals["ridge"])) if "ridge" in cond_vals and cond_vals["ridge"] else float("nan")
    summary: dict[str, dict] = {}
    for cond in active_conditions:
        if not cond_vals[cond]:
            continue
        mean_r = float(np.mean(cond_vals[cond]))
        std_r = float(np.std(cond_vals[cond]))
        summary[cond] = {
            "per_drug_r_mean": mean_r,
            "per_drug_r_std": std_r,
            "delta_vs_ridge": round(mean_r - ridge_mean, 4) if not np.isnan(ridge_mean) else None,
        }
        logger.info("  %-8s  %.4f ± %.4f", cond, mean_r, std_r)

    pred_corr_summary: dict[str, dict] = {}
    for cond, vals in corr_vals.items():
        if vals:
            mc = float(np.mean(vals))
            pred_corr_summary[cond] = {"mean": mc, "values": vals}
            logger.info("  pred_corr(ridge, %-8s): %.4f", cond, mc)

    output: dict = {
        "condition": args.condition,
        "summary": summary,
        "fold_results": all_folds,
    }
    if pred_corr_summary:
        output["pred_corr"] = pred_corr_summary

    if len(active_conditions) > 1 and "xgboost" in summary and "mlp" in summary and not np.isnan(ridge_mean):
        best_nonlinear = max(
            summary["xgboost"]["per_drug_r_mean"],
            summary["mlp"]["per_drug_r_mean"],
        )
        if best_nonlinear - ridge_mean > 0.01:
            verdict = f"Nonlinear model exceeds Ridge by Δ={best_nonlinear - ridge_mean:.3f} — ceiling is Ridge-limited."
        else:
            verdict = "Nonlinear models Δ≤0.01 — ceiling is not Ridge-specific; drug-blind problem is fundamentally hard."
        logger.info("Verdict: %s", verdict)
        output["verdict"] = verdict

    results_path.write_text(json.dumps(output, indent=2))
    logger.info("Done. Results: %s", results_path)


if __name__ == "__main__":
    main()
