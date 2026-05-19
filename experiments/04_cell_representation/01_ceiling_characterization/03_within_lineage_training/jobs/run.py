"""03_within_lineage_training: train Ridge on single lineage only, evaluate drug-blind.

Tests whether cross-lineage variance drives the pan-cancer per-drug r=0.631.
If the model learned "hematologic cells are more sensitive on average", restricting
both training and test to one lineage should give dramatically lower r.
Expected (null): within-lineage r ≈ pan-cancer per-lineage r (Δ ≤ 0.05).

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
MIN_TRAIN_CELLS = 20  # skip lineage-fold if fewer training cells

LINEAGE_MAP = {
    "HAEMATOPOIETIC_AND_LYMPHOID_TISSUE": "Hematologic",
    "LUNG": "Lung",
    "SKIN": "Skin",
    "CENTRAL_NERVOUS_SYSTEM": "CNS",
    "BREAST": "Breast",
    "LARGE_INTESTINE": "Colorectal",
}
LINEAGES = sorted(set(LINEAGE_MAP.values()))


def build_depmap_to_lineage(data_dir: Path) -> dict[str, str]:
    cl_idx = pd.read_parquet(data_dir / "cell_line_index.parquet")
    result: dict[str, str] = {}
    for depmap_id, row in cl_idx.iterrows():
        parts = str(row.get("ccle_name", "")).split("_", 1)
        tissue = parts[1] if len(parts) > 1 else "UNKNOWN"
        result[str(depmap_id)] = LINEAGE_MAP.get(tissue, "Other")
    return result


def run_fold(
    fold_i: int,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    available_cells: set[str],
    name_to_depmap: dict[str, str],
    depmap_to_lineage: dict[str, str],
) -> dict:
    train_df, test_df = load_paso_pairs(
        PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
    )
    if len(train_df) == 0 or len(test_df) == 0:
        return {}

    # Pan-cancer reference (all lineages)
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

    sc = safe_fit_scaler(cell_feat[tr_idx])
    ridge = Ridge(alpha=1.0)
    ridge.fit(sc.transform(cell_feat[tr_idx]), y_train)
    preds = ridge.predict(sc.transform(cell_feat[te_idx])).astype(np.float32)

    pancancer_r = float(mean_per_drug_r(preds, y_test, test_df["drug_name"].values,
                                        min_cells=MIN_CELLS))
    logger.info("  fold %d pancancer: %.4f (%d drugs)", fold_i, pancancer_r,
                test_df["drug_name"].nunique())

    # Per-lineage within-lineage training
    lineage_results: dict[str, dict] = {}
    for lin in LINEAGES:
        lin_train_cells = sorted(
            c for c in train_df["depmap_id"].unique() if depmap_to_lineage.get(c) == lin
        )
        lin_test_cells = sorted(
            c for c in test_df["depmap_id"].unique() if depmap_to_lineage.get(c) == lin
        )

        if len(lin_train_cells) < MIN_TRAIN_CELLS:
            logger.info("  fold %d [%-12s]: skip (only %d train cells)",
                        fold_i, lin, len(lin_train_cells))
            lineage_results[lin] = {
                "per_drug_r": float("nan"),
                "n_train_cells": len(lin_train_cells),
                "n_test_cells": len(lin_test_cells),
                "n_drugs": 0,
                "skipped": True,
            }
            continue

        lin_train_df = train_df[train_df["depmap_id"].isin(set(lin_train_cells))].copy()
        lin_test_df = test_df[test_df["depmap_id"].isin(set(lin_test_cells))].copy()

        if len(lin_test_df) == 0:
            lineage_results[lin] = {
                "per_drug_r": float("nan"),
                "n_train_cells": len(lin_train_cells),
                "n_test_cells": 0,
                "n_drugs": 0,
                "skipped": True,
            }
            continue

        lin_all_cells = sorted(set(lin_train_cells) | set(lin_test_cells))
        lin_cell_to_row = {c: i for i, c in enumerate(lin_all_cells)}
        lin_rna = rna.loc[lin_all_cells].values.astype(np.float32)
        lin_mut = mutations.loc[lin_all_cells].values.astype(np.float32)
        lin_train_rows = np.array([lin_cell_to_row[c] for c in lin_train_cells], dtype=np.int32)

        lin_rna_pca, lin_mut_pca = compress_cell(
            lin_rna, lin_mut, lin_train_rows, rna_dim=RNA_DIM, mut_dim=MUT_DIM
        )
        lin_feat = np.concatenate([lin_rna_pca, lin_mut_pca], axis=1)

        lin_tr_idx = np.array(
            [lin_cell_to_row[c] for c in lin_train_df["depmap_id"]], dtype=np.int32
        )
        lin_te_idx = np.array(
            [lin_cell_to_row[c] for c in lin_test_df["depmap_id"]], dtype=np.int32
        )

        lin_sc = safe_fit_scaler(lin_feat[lin_tr_idx])
        lin_ridge = Ridge(alpha=1.0)
        lin_ridge.fit(lin_sc.transform(lin_feat[lin_tr_idx]),
                      lin_train_df["ln_ic50"].values.astype(np.float32))
        lin_preds = lin_ridge.predict(
            lin_sc.transform(lin_feat[lin_te_idx])
        ).astype(np.float32)

        lin_y_test = lin_test_df["ln_ic50"].values.astype(np.float32)
        lin_drugs = lin_test_df["drug_name"].values
        lin_r = float(mean_per_drug_r(lin_preds, lin_y_test, lin_drugs, min_cells=MIN_CELLS))
        n_drugs = int(np.unique(lin_drugs).size)

        logger.info("  fold %d [%-12s]: within_r=%.4f (%d train, %d drugs)",
                    fold_i, lin, lin_r, len(lin_train_cells), n_drugs)
        lineage_results[lin] = {
            "per_drug_r": lin_r,
            "n_train_cells": len(lin_train_cells),
            "n_test_cells": len(lin_test_cells),
            "n_drugs": n_drugs,
            "skipped": False,
        }

    return {"fold": fold_i, "pancancer_r": pancancer_r, "by_lineage": lineage_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 fold only")
    args = parser.parse_args()
    k_folds = 1 if args.smoke else K_FOLDS

    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(logs_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("03_within_lineage_training: within-lineage drug-blind per-drug r%s",
                " [SMOKE]" if args.smoke else "")

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    available_cells = set(rna.index) & set(mutations.index)
    logger.info("RNA: %s  Mutations: %s  available: %d cells",
                rna.shape, mutations.shape, len(available_cells))

    name_to_depmap = load_cell_line_index(DATA_DIR)
    depmap_to_lineage = build_depmap_to_lineage(DATA_DIR)

    lin_counts: dict[str, int] = {}
    for dep in available_cells:
        lin = depmap_to_lineage.get(dep, "Other")
        lin_counts[lin] = lin_counts.get(lin, 0) + 1
    for lin, cnt in sorted(lin_counts.items(), key=lambda x: -x[1]):
        logger.info("  lineage %-20s: %d cells", lin, cnt)

    results_path = report_dir / "results.json"
    all_folds: list[dict] = []

    for fold_i in range(k_folds):
        logger.info("=== Fold %d/%d ===", fold_i + 1, k_folds)
        fold_res = run_fold(fold_i, rna, mutations, available_cells,
                            name_to_depmap, depmap_to_lineage)
        if fold_res:
            all_folds.append(fold_res)
            results_path.write_text(json.dumps({"fold_results": all_folds}, indent=2))

    # Summary
    logger.info("=" * 70)
    pancancer_vals = [r["pancancer_r"] for r in all_folds]
    logger.info("Pan-cancer: %.4f ± %.4f", float(np.mean(pancancer_vals)),
                float(np.std(pancancer_vals)))

    lineage_summary: dict[str, list[float]] = {}
    for r in all_folds:
        for lin, res in r["by_lineage"].items():
            if not res["skipped"] and not np.isnan(res["per_drug_r"]) and res["n_drugs"] > 0:
                lineage_summary.setdefault(lin, []).append(res["per_drug_r"])

    for lin in LINEAGES:
        if lin in lineage_summary:
            vals = lineage_summary[lin]
            logger.info("  %-15s within_r=%.4f ± %.4f (%d valid folds)",
                        lin, float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            logger.info("  %-15s no valid folds", lin)

    # Final output with aggregated stats
    final = {
        "pancancer_overall": {
            "mean": float(np.mean(pancancer_vals)),
            "std": float(np.std(pancancer_vals)),
        },
        "by_lineage": {
            lin: {
                "within_lineage_folds": [
                    r["by_lineage"].get(lin, {}) for r in all_folds
                    if lin in r["by_lineage"]
                ],
                "within_lineage_mean": float(np.mean(lineage_summary[lin]))
                    if lin in lineage_summary else float("nan"),
                "within_lineage_std": float(np.std(lineage_summary[lin]))
                    if lin in lineage_summary else float("nan"),
                "n_valid_folds": len(lineage_summary.get(lin, [])),
            }
            for lin in LINEAGES
        },
        "fold_results": all_folds,
    }
    results_path.write_text(json.dumps(final, indent=2))
    logger.info("Done. Results: %s", results_path)


if __name__ == "__main__":
    main()
