"""03_ranking_loss: drug-standardized Ridge vs MSE Ridge, PASO drug-blind CV.

Tests whether the per-drug r=0.631 ceiling is an artifact of MSE loss mismatch.
Drug-standardized Ridge normalizes each drug's IC50 to z-scores before training,
making MSE equivalent to within-drug ranking loss (Pearson r maximization).
Expected (null): standardized r ≈ raw r (Δ ≤ 0.005) — Ridge's shared linear
function already maximizes within-drug ranking regardless of target scale.

Usage:
  python run.py                           # all conditions → results.json
  python run.py --condition ridge_mse     # single → results_ridge_mse.json
  python run.py --smoke                   # 1 fold only

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
from sklearn.linear_model import Ridge

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
MIN_DRUG_STD = 0.01

# Conditions: (name, alpha)
ALL_CONDITIONS: list[tuple[str, float]] = [
    ("ridge_mse", 1.0),
    ("ridge_rank", 1.0),
    ("ridge_rank_01", 0.1),
    ("ridge_rank_10", 10.0),
]
ALL_CONDITION_NAMES = [n for n, _ in ALL_CONDITIONS]
ALPHA_BY_NAME = {n: a for n, a in ALL_CONDITIONS}


def drug_standardize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize ln_ic50 to within-drug z-scores using training drug statistics."""
    drug_stats = train_df.groupby("drug_name")["ln_ic50"].agg(["mean", "std"]).copy()
    drug_stats["std"] = drug_stats["std"].clip(lower=MIN_DRUG_STD)

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for drug, stats_row in drug_stats.iterrows():
            mask = df["drug_name"] == drug
            df.loc[mask, "ln_ic50"] = (
                (df.loc[mask, "ln_ic50"] - stats_row["mean"]) / stats_row["std"]
            )
        return df

    return normalize(train_df), normalize(test_df)


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
    y_test_raw = test_df["ln_ic50"].values.astype(np.float32)
    drug_names = test_df["drug_name"].values

    sc = safe_fit_scaler(cell_feat[tr_idx])
    X_train = sc.transform(cell_feat[tr_idx])
    X_test = sc.transform(cell_feat[te_idx])

    # Drug-standardized training targets (computed once if any rank condition active)
    needs_rank = any(n.startswith("ridge_rank") for n in active_conditions)
    std_train_df = drug_standardize(train_df, test_df)[0] if needs_rank else None

    fold_result: dict = {"fold": fold_i}

    for name in active_conditions:
        alpha = ALPHA_BY_NAME[name]
        is_rank = name.startswith("ridge_rank")
        y_tr = (std_train_df["ln_ic50"].values if is_rank  # type: ignore[union-attr]
                else train_df["ln_ic50"].values)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_tr.astype(np.float32))
        preds = ridge.predict(X_test).astype(np.float32)

        r = float(mean_per_drug_r(preds, y_test_raw, drug_names, min_cells=MIN_CELLS))
        logger.info("  fold %d [%-18s]: per_drug_r=%.4f", fold_i, name, r)
        fold_result[name] = {"per_drug_r": r}

    return fold_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 fold only")
    parser.add_argument("--condition", type=str, choices=ALL_CONDITION_NAMES, default=None,
                        help="Run a single condition (writes results_<condition>.json)")
    args = parser.parse_args()
    k_folds = 1 if args.smoke else K_FOLDS
    active_conditions = [args.condition] if args.condition else ALL_CONDITION_NAMES

    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(logs_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("03_ranking_loss: drug-standardized Ridge vs MSE Ridge%s%s",
                f" [cond={args.condition}]" if args.condition else "",
                " [SMOKE]" if args.smoke else "")

    if args.condition:
        results_path = report_dir / f"results_{args.condition}.json"
    else:
        results_path = report_dir / "results.json"

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    available_cells = set(rna.index) & set(mutations.index)
    name_to_depmap = load_cell_line_index(DATA_DIR)
    logger.info("available cells: %d", len(available_cells))

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
    cond_vals: dict[str, list[float]] = {name: [] for name in active_conditions}
    for r in all_folds:
        for name in active_conditions:
            if name in r:
                cond_vals[name].append(r[name]["per_drug_r"])

    mse_mean = float(np.mean(cond_vals["ridge_mse"])) if "ridge_mse" in cond_vals and cond_vals["ridge_mse"] else float("nan")
    summary: dict[str, dict] = {}
    for name in active_conditions:
        if not cond_vals[name]:
            continue
        mean_r = float(np.mean(cond_vals[name]))
        std_r = float(np.std(cond_vals[name]))
        summary[name] = {
            "per_drug_r_mean": mean_r,
            "per_drug_r_std": std_r,
            "delta": round(mean_r - mse_mean, 4) if not np.isnan(mse_mean) else None,
        }
        logger.info("  %-20s  %.4f ± %.4f", name, mean_r, std_r)

    output: dict = {
        "condition": args.condition,
        "summary": summary,
        "fold_results": all_folds,
    }

    if len(active_conditions) > 1 and "ridge_mse" in summary and "ridge_rank" in summary:
        rank_delta = summary["ridge_rank"]["delta"] or 0.0
        if abs(rank_delta) <= 0.005:
            verdict = (
                f"Drug-standardized Δ={rank_delta:.4f} ≤ 0.005 — "
                "MSE and ranking objectives are equivalent for no-drug-feature Ridge; "
                "ceiling is not MSE-loss-specific."
            )
        else:
            verdict = (
                f"Drug-standardized Δ={rank_delta:.4f} > 0.005 — "
                "ranking loss changes per-drug r; revise to 'MSE-loss ceiling'."
            )
        logger.info("Verdict: %s", verdict)
        output["verdict"] = verdict

    results_path.write_text(json.dumps(output, indent=2))
    logger.info("Done. Results: %s", results_path)


if __name__ == "__main__":
    main()
