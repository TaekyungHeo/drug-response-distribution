"""01_learning_curve: is the drug-blind r=0.631 ceiling data-limited?

For each training cell fraction (0.1→1.0), runs PASO 10-fold drug-blind CV with Ridge
and reports per-drug r. A plateau before fraction=1.0 means the ceiling is
information-theoretic; a rising curve at fraction=1.0 means it is data-limited.

Conditions (all use Ridge(α=1.0), PASO 10-fold drug-blind CV, per-drug r):
  fraction ∈ [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

Usage:
  python run.py                       # all fractions → results.json
  python run.py --fraction 0.1        # single fraction → results_frac_0_1.json
  python run.py --smoke               # 1 fold only

Output: report/data/results.json  (or results_frac_<f>.json)
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
RIDGE_ALPHA = 1.0
RNA_DIM = 550
MUT_DIM = 200
MIN_CELLS_EVAL = 5

ALL_FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def fraction_key(frac: float) -> str:
    """Canonical key for a fraction value."""
    return f"frac_{frac:.1f}".replace(".", "_")


def run_fraction(
    fraction: float,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    available_cells: set[str],
    name_to_depmap: dict[str, str],
    k_folds: int,
) -> dict:
    """Run PASO k_folds-fold drug-blind CV with training cells subsampled to `fraction`."""
    logger.info("=== Fraction: %.1f ===", fraction)
    folds_out = []

    for fold_i in range(k_folds):
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
        )
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("  fold %d: empty — skip", fold_i)
            continue

        # Subsample training cell lines (test cells unchanged)
        train_cells_all = sorted(train_df["depmap_id"].unique())
        rng = np.random.default_rng(42 + fold_i)
        n_keep = max(1, int(len(train_cells_all) * fraction))
        train_cells_sub = sorted(rng.choice(train_cells_all, size=n_keep, replace=False))
        train_df_sub = train_df[train_df["depmap_id"].isin(set(train_cells_sub))].copy()

        all_cells = sorted(set(train_cells_sub) | set(test_df["depmap_id"]))
        cell_to_row = {c: i for i, c in enumerate(all_cells)}
        train_rows = np.array([cell_to_row[c] for c in train_cells_sub], dtype=np.int32)

        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_rows,
                                         rna_dim=RNA_DIM, mut_dim=MUT_DIM)
        cell_feat = np.concatenate([rna_pca, mut_pca], axis=1)

        tr_idx = np.array([cell_to_row[c] for c in train_df_sub["depmap_id"]], dtype=np.int32)
        te_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
        y_train = train_df_sub["ln_ic50"].values.astype(np.float32)
        y_test = test_df["ln_ic50"].values.astype(np.float32)

        sc = safe_fit_scaler(cell_feat[tr_idx])
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(sc.transform(cell_feat[tr_idx]), y_train)
        preds = ridge.predict(sc.transform(cell_feat[te_idx])).astype(np.float32)

        per_dr = mean_per_drug_r(preds, y_test, test_df["drug_name"].values,
                                 min_cells=MIN_CELLS_EVAL)
        logger.info(
            "  fold %d: n_train_cells=%d n_train_pairs=%d per_drug_r=%.4f",
            fold_i, len(train_cells_sub), len(train_df_sub), per_dr,
        )
        folds_out.append({"per_drug_r": per_dr, "n_train_cells": len(train_cells_sub)})

    per_drug_rs = [f["per_drug_r"] for f in folds_out]
    return {
        "fraction": fraction,
        "key": fraction_key(fraction),
        "folds": folds_out,
        "per_drug_r_mean": float(np.mean(per_drug_rs)) if per_drug_rs else float("nan"),
        "per_drug_r_std": float(np.std(per_drug_rs)) if per_drug_rs else float("nan"),
        "n_train_cells_mean": float(np.mean([f["n_train_cells"] for f in folds_out]))
                              if folds_out else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 fold only")
    parser.add_argument("--fraction", type=float, choices=ALL_FRACTIONS, default=None,
                        help="Run a single fraction (writes results_frac_<f>.json)")
    args = parser.parse_args()
    k_folds = 1 if args.smoke else K_FOLDS
    active_fractions = [args.fraction] if args.fraction is not None else ALL_FRACTIONS

    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(logs_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("01_learning_curve: data sufficiency test on PASO drug-blind per-drug r%s%s",
                f" [frac={args.fraction}]" if args.fraction is not None else "",
                " [SMOKE]" if args.smoke else "")

    if args.fraction is not None:
        results_path = report_dir / f"results_{fraction_key(args.fraction)}.json"
    else:
        results_path = report_dir / "results.json"

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    available_cells = set(rna.index) & set(mutations.index)
    logger.info("RNA: %s  Mutations: %s  available cells: %d",
                rna.shape, mutations.shape, len(available_cells))

    name_to_depmap = load_cell_line_index(DATA_DIR)

    results: dict[str, dict] = {}
    for frac in active_fractions:
        key = fraction_key(frac)
        res = run_fraction(frac, rna, mutations, available_cells, name_to_depmap, k_folds)
        results[key] = res

    logger.info("=" * 60)
    logger.info("%-12s  %10s  %6s  %10s", "Fraction", "per-drug r", "±std", "n_cells")
    for _key, res in results.items():
        logger.info("%-12s  %10.4f  %6.4f  %10.1f",
                    f"{res['fraction']:.1f}", res["per_drug_r_mean"],
                    res["per_drug_r_std"], res["n_train_cells_mean"])

    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
