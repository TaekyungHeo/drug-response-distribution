"""01_split_ceilings: drug-blind vs cell-blind per-drug r with Ridge.

Establishes cell-blind per-drug r as a §Discussion anchor:
  - Drug-blind: PASO 5-fold drug-blind CV (233 drugs), Ridge, no drug features
  - Cell-blind: 5-fold CV on cells (same 233 drugs), Ridge, no drug features
  - Cheat predictor: drug-mean baseline to validate global r inflation

Expected:
  drug_blind per-drug r ≈ 0.631 ± 0.023  (matches 02_representation_ablation baseline)
  cell_blind per-drug r ≈ 0.46 ± 0.03
  cheat_predictor global r ≈ 0.79

Output: report/data/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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

K_FOLDS_DRUG = 10   # PASO drug-blind CV folds (matches canonical 02_representation_ablation)
K_FOLDS_CELL = 5    # cell-blind CV folds (tractable; drug-blind folds fixed by PASO)
RIDGE_ALPHA = 1.0
RNA_DIM = 550
MUT_DIM = 200
MIN_CELLS_EVAL = 5


def build_cell_features(
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    all_cells: list[str],
    train_cells: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Return (cell_feat, cell_to_row) with PCA fit on train_cells only."""
    cell_to_row = {c: i for i, c in enumerate(all_cells)}
    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    train_rows = np.array([cell_to_row[c] for c in train_cells], dtype=np.int32)
    rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_rows,
                                     rna_dim=RNA_DIM, mut_dim=MUT_DIM)
    return np.concatenate([rna_pca, mut_pca], axis=1), cell_to_row


def fit_predict(
    cell_feat: np.ndarray,
    cell_to_row: dict[str, int],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge fit on train_df, predict on test_df. Returns (preds, y_test, test_drug_names)."""
    tr_cell_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
    te_cell_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)

    X_train = cell_feat[tr_cell_idx]
    X_test = cell_feat[te_cell_idx]
    y_train = train_df["ln_ic50"].values.astype(np.float32)
    y_test = test_df["ln_ic50"].values.astype(np.float32)

    sc = safe_fit_scaler(X_train)
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(sc.transform(X_train), y_train)
    preds = ridge.predict(sc.transform(X_test)).astype(np.float32)
    return preds, y_test, test_df["drug_name"].values


def run_drug_blind(
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    available_cells: set[str],
    name_to_depmap: dict[str, str],
    k_folds: int = K_FOLDS_DRUG,
) -> dict:
    """PASO k_folds-fold drug-blind CV."""
    logger.info("=== Drug-blind: PASO %d-fold ===", k_folds)
    folds_out = []

    for fold_i in range(k_folds):
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
        )
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("fold %d: empty train or test — skip", fold_i)
            continue

        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        train_cells = sorted(train_df["depmap_id"].unique())
        cell_feat, cell_to_row = build_cell_features(rna, mutations, all_cells, train_cells)

        preds, y_test, drug_names = fit_predict(cell_feat, cell_to_row, train_df, test_df)
        per_drug_r = mean_per_drug_r(preds, y_test, drug_names, min_cells=MIN_CELLS_EVAL)
        global_r = float(pearsonr(preds, y_test).statistic)
        n_drugs = len(np.unique(drug_names))

        cell_cheat = run_cell_mean_cheat(train_df, test_df, y_test, drug_names)

        logger.info("  fold %d: per_drug_r=%.4f  global_r=%.4f  n_drugs=%d",
                    fold_i, per_drug_r, global_r, n_drugs)
        folds_out.append({
            "per_drug_r": per_drug_r,
            "global_r": global_r,
            "n_drugs": n_drugs,
            "cell_mean_cheat_per_drug_r": cell_cheat["per_drug_r"],
            "cell_mean_cheat_global_r": cell_cheat["global_r"],
        })

    per_drug_rs = [f["per_drug_r"] for f in folds_out]
    cell_cheat_rs = [f["cell_mean_cheat_per_drug_r"] for f in folds_out]
    return {
        "folds": folds_out,
        "per_drug_r_mean": float(np.mean(per_drug_rs)),
        "per_drug_r_std": float(np.std(per_drug_rs)),
        "global_r_mean": float(np.mean([f["global_r"] for f in folds_out])),
        "cell_mean_cheat_per_drug_r_mean": float(np.mean(cell_cheat_rs)),
        "cell_mean_cheat_per_drug_r_std": float(np.std(cell_cheat_rs)),
    }


def run_cell_blind(
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    dr: pd.DataFrame,
    k_folds: int = K_FOLDS_CELL,
    k_iters: int | None = None,
) -> dict:
    """k_folds-fold CV on cell lines, same 233 PASO drugs.

    k_iters: number of fold iterations to run (default: k_folds).
    Smoke mode passes k_iters=1 while keeping k_folds=K_FOLDS_CELL to preserve
    meaningful train/test splits.
    """
    n_iters = k_iters if k_iters is not None else k_folds
    logger.info("=== Cell-blind: %d-fold split, %d iters ===", k_folds, n_iters)
    all_cells = sorted(dr["depmap_id"].unique())
    rng = np.random.default_rng(42)
    shuffled_cells = rng.permutation(all_cells)
    cell_folds = np.array_split(shuffled_cells, k_folds)

    folds_out = []
    cheat_fold0: dict | None = None  # save fold 0 for cheat predictor

    for fold_i in range(n_iters):
        test_cells = set(cell_folds[fold_i])
        train_cells_list = [c for i, f in enumerate(cell_folds) if i != fold_i for c in f]
        train_cells = sorted(train_cells_list)

        train_df = dr[dr["depmap_id"].isin(set(train_cells))].copy()
        test_df = dr[dr["depmap_id"].isin(test_cells)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("fold %d: empty split — skip", fold_i)
            continue

        all_cells_fold = sorted(set(train_cells) | test_cells)
        cell_feat, cell_to_row = build_cell_features(rna, mutations, all_cells_fold, train_cells)

        preds, y_test, drug_names = fit_predict(cell_feat, cell_to_row, train_df, test_df)
        per_drug_r = mean_per_drug_r(preds, y_test, drug_names, min_cells=MIN_CELLS_EVAL)
        global_r = float(pearsonr(preds, y_test).statistic)
        n_drugs = len(np.unique(drug_names))

        logger.info("  fold %d: per_drug_r=%.4f  global_r=%.4f  n_drugs=%d",
                    fold_i, per_drug_r, global_r, n_drugs)
        folds_out.append({"per_drug_r": per_drug_r, "global_r": global_r, "n_drugs": n_drugs})

        if fold_i == 0:
            cheat_fold0 = {
                "train_df": train_df,
                "y_test": y_test,
                "drug_names": drug_names,
            }

    per_drug_rs = [f["per_drug_r"] for f in folds_out]
    return {
        "folds": folds_out,
        "per_drug_r_mean": float(np.mean(per_drug_rs)),
        "per_drug_r_std": float(np.std(per_drug_rs)),
        "global_r_mean": float(np.mean([f["global_r"] for f in folds_out])),
        "_cheat_fold0": cheat_fold0,
    }


def run_drug_mean_cheat(
    train_df: pd.DataFrame,
    y_test: np.ndarray,
    drug_names: np.ndarray,
) -> dict:
    """Drug-mean baseline: predict each test pair as training mean for that drug.

    Global r captures between-drug variance only — validates Sealfon-style inflation.
    per_drug_r ≈ 0 expected (constant predictions per drug → zero within-drug variance).
    """
    drug_means_dict: dict[str, float] = train_df.groupby("drug_name")["ln_ic50"].mean().to_dict()
    global_train_mean = float(train_df["ln_ic50"].mean())

    preds = np.array([
        drug_means_dict.get(d, global_train_mean)
        for d in drug_names
    ], dtype=np.float32)

    global_r = float(pearsonr(preds, y_test).statistic)
    per_drug_r_val = mean_per_drug_r(preds, y_test, drug_names, min_cells=MIN_CELLS_EVAL)
    logger.info("  drug_mean_cheat: global_r=%.4f  per_drug_r=%.4f", global_r, per_drug_r_val)
    return {
        "global_r": global_r,
        "per_drug_r": per_drug_r_val,
        "note": "drug-mean from training; per_drug_r~0 expected (constant per drug)",
    }


def run_cell_mean_cheat(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    drug_names: np.ndarray,
) -> dict:
    """Cell-mean baseline: predict each (cell, drug) pair as that cell's mean IC50 across train drugs.

    Critical diagnostic: if per_drug_r ≈ drug_blind Ridge r, then Ridge is learning
    cell-toxicity-score (which cells are globally sensitive), not drug-specific signal.
    If per_drug_r << Ridge, the model captures genuine drug-specific within-cell variation.
    """
    cell_means_dict: dict[str, float] = train_df.groupby("depmap_id")["ln_ic50"].mean().to_dict()
    global_train_mean = float(train_df["ln_ic50"].mean())

    preds = np.array([
        cell_means_dict.get(c, global_train_mean)
        for c in test_df["depmap_id"]
    ], dtype=np.float32)

    global_r = float(pearsonr(preds, y_test).statistic)
    per_drug_r_val = mean_per_drug_r(preds, y_test, drug_names, min_cells=MIN_CELLS_EVAL)
    logger.info("  cell_mean_cheat: global_r=%.4f  per_drug_r=%.4f", global_r, per_drug_r_val)
    return {
        "global_r": global_r,
        "per_drug_r": per_drug_r_val,
        "note": "cell-mean across train drugs; if per_drug_r≈Ridge then Ridge learns toxicity-score only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 fold only")
    args = parser.parse_args()
    k_folds_drug = 1 if args.smoke else K_FOLDS_DRUG
    k_folds_cell = 1 if args.smoke else K_FOLDS_CELL

    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(logs_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("01_split_ceilings: drug-blind vs cell-blind Ridge per-drug r%s",
                " [SMOKE]" if args.smoke else "")

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    available_cells = set(rna.index) & set(mutations.index)
    logger.info("RNA: %s  Mutations: %s  available cells: %d",
                rna.shape, mutations.shape, len(available_cells))

    name_to_depmap = load_cell_line_index(DATA_DIR)

    # Build PASO drug set (233 drugs)
    paso_drug_set: set[str] = set()
    for fold_i in range(10):
        tr = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{fold_i}.csv")
        te = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{fold_i}.csv")
        paso_drug_set |= set(tr["drug"].unique()) | set(te["drug"].unique())
    logger.info("PASO drug set: %d drugs", len(paso_drug_set))

    # Load drug response for cell-blind (all PASO drugs, available cells)
    dr_raw = pd.read_parquet(DATA_DIR / "drug_response.parquet")
    dr = dr_raw[
        dr_raw["depmap_id"].isin(available_cells) & dr_raw["drug_name"].isin(paso_drug_set)
    ].copy().reset_index(drop=True)
    logger.info("Cell-blind dataset: %d pairs, %d drugs, %d cells",
                len(dr), dr["drug_name"].nunique(), dr["depmap_id"].nunique())

    # ── Drug-blind ──
    drug_blind_results = run_drug_blind(rna, mutations, available_cells, name_to_depmap,
                                        k_folds=k_folds_drug)

    # ── Cell-blind ──
    # Keep K_FOLDS_CELL for the split; only run k_folds_cell iterations (smoke: 1)
    cell_blind_results = run_cell_blind(rna, mutations, dr,
                                        k_folds=K_FOLDS_CELL, k_iters=k_folds_cell)
    cheat_data = cell_blind_results.pop("_cheat_fold0")

    # ── Drug-mean cheat predictor (fold 0 cell-blind) ──
    cheat_results = run_drug_mean_cheat(
        cheat_data["train_df"],
        cheat_data["y_test"],
        cheat_data["drug_names"],
    )

    # Summary
    logger.info("=" * 70)
    logger.info("%-22s  %10s  %8s", "Split", "per-drug r", "global r")
    logger.info("%-22s  %10.4f  %8.4f", "drug_blind",
                drug_blind_results["per_drug_r_mean"], drug_blind_results["global_r_mean"])
    logger.info("%-22s  %10.4f  (cell-mean cheat, drug-blind)",
                "cell_mean_cheat", drug_blind_results["cell_mean_cheat_per_drug_r_mean"])
    logger.info("%-22s  %10.4f  %8.4f", "cell_blind",
                cell_blind_results["per_drug_r_mean"], cell_blind_results["global_r_mean"])
    logger.info("%-22s  %10.4f  %8.4f  (drug-mean cheat, fold-0 cell-blind)",
                "drug_mean_cheat", cheat_results["per_drug_r"], cheat_results["global_r"])
    gap = cell_blind_results["per_drug_r_mean"] - drug_blind_results["per_drug_r_mean"]
    cheat_gap = drug_blind_results["cell_mean_cheat_per_drug_r_mean"] - drug_blind_results["per_drug_r_mean"]
    logger.info("Gap (cell_blind − drug_blind): %.4f", gap)
    logger.info("Cell-mean cheat gap vs drug_blind: %.4f  (≈0 → toxicity-score; <<0 → drug-specific signal)",
                cheat_gap)

    output = {
        "drug_blind": drug_blind_results,
        "cell_blind": cell_blind_results,
        "drug_mean_cheat_predictor": cheat_results,
    }
    out_path = report_dir / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
