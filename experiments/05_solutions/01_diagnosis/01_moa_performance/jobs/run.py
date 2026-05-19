"""Per-MoA per-drug r under all-drug Ridge training.

Runs Ridge(alpha=1.0) with RNA PCA(550) + mutation PCA(200) cell features
across PASO 10-fold drug-blind CV, computes per-drug Pearson r, then groups
results by MoA (Target Pathway) and reports per-MoA mean/std.

CLI:
  --smoke    run only 2 folds (fast check)

Output: report/data/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.evaluation.per_drug import per_drug_r
from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell
from src.utils.solutions import group_drugs_by_moa, load_moa_annotations

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
RIDGE_ALPHA = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-MoA performance diagnosis")
    parser.add_argument("--smoke", action="store_true", help="Run only 2 folds")
    args = parser.parse_args()

    n_folds = 2 if args.smoke else K_FOLDS
    logger.info("01_moa_performance | ROOT=%s | folds=%d", ROOT, n_folds)

    # ---- Load omics ----
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    available_cells = set(rna.index) & set(mutations.index)

    # ---- Build drug index from all PASO folds ----
    all_drugs: set[str] = set()
    for k in range(K_FOLDS):
        tr = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{k}.csv")
        te = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{k}.csv")
        all_drugs |= set(tr["drug"].unique()) | set(te["drug"].unique())
    drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(sorted(all_drugs))}
    logger.info("PASO drug set: %d drugs", len(drug_to_idx))

    # ---- MoA annotations ----
    moa = load_moa_annotations()
    logger.info("MoA annotations loaded: %d drugs annotated", len(moa))

    # ---- Run folds, pool per-drug r ----
    # Drug-blind: each drug appears in exactly one test fold.
    # We pool per-drug r across folds to get one r per drug.
    pooled_per_drug_r: dict[str, float] = {}

    for fold_i in range(n_folds):
        t0 = datetime.now()
        logger.info("Fold %d/%d started at %s", fold_i, n_folds, t0.strftime("%H:%M:%S"))

        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
        )
        train_df = pd.DataFrame(train_df[train_df["drug_name"].isin(drug_to_idx)])
        test_df = pd.DataFrame(test_df[test_df["drug_name"].isin(drug_to_idx)])

        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty train or test — skipping", fold_i)
            continue

        # Cell features
        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        cell_to_row = {c: i for i, c in enumerate(all_cells)}
        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        train_cell_rows = np.array(
            [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
        )
        train_cell_set = np.unique(train_cell_rows)
        rna_c, mut_c = compress_cell(rna_arr, mut_arr, train_cell_set)
        cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

        # Build pair matrices (cell features only — no drug features)
        train_rows_c = np.array(
            [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
        )
        test_rows_c = np.array(
            [cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32
        )
        X_train = cell_feat[train_rows_c]
        y_train = train_df["ln_ic50"].values.astype(np.float32)
        X_test = cell_feat[test_rows_c]
        y_test = test_df["ln_ic50"].values.astype(np.float32)
        drug_names_test = test_df["drug_name"].values

        logger.info(
            "  Fold %d: train=%d test=%d n_features=%d",
            fold_i, len(y_train), len(y_test), X_train.shape[1],
        )

        # Fit Ridge
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
        preds = model.predict(X_test.astype(np.float64)).astype(np.float32)

        # Per-drug r
        fold_per_drug = per_drug_r(preds, y_test, drug_names_test, min_cells=5)
        pooled_per_drug_r.update(fold_per_drug)

        t1 = datetime.now()
        elapsed = (t1 - t0).total_seconds()
        logger.info(
            "  Fold %d: %d drugs evaluated, mean_r=%.4f, elapsed=%.1fs",
            fold_i,
            len(fold_per_drug),
            float(np.mean(list(fold_per_drug.values()))) if fold_per_drug else float("nan"),
            elapsed,
        )

    # ---- Aggregate ----
    all_rs = list(pooled_per_drug_r.values())
    overall_mean_r = float(np.mean(all_rs)) if all_rs else float("nan")
    logger.info(
        "Overall: %d drugs, grand mean per-drug r = %.4f",
        len(pooled_per_drug_r), overall_mean_r,
    )

    # ---- Group by MoA ----
    all_drug_names = list(pooled_per_drug_r.keys())
    moa_groups = group_drugs_by_moa(all_drug_names, moa)

    per_moa: list[dict] = []
    for pathway, drugs in sorted(moa_groups.items()):
        rs = [pooled_per_drug_r[d] for d in drugs if d in pooled_per_drug_r]
        if not rs:
            continue
        per_moa.append({
            "moa": pathway,
            "mean_r": round(float(np.mean(rs)), 6),
            "std_r": round(float(np.std(rs)), 6),
            "n_drugs": len(rs),
            "drugs": sorted(drugs),
        })
    per_moa.sort(key=lambda x: x["mean_r"], reverse=True)

    # Drugs without MoA annotation
    annotated = set()
    for g in moa_groups.values():
        annotated.update(g)
    unannotated_drugs = [d for d in all_drug_names if d not in annotated]
    unannotated_rs = [pooled_per_drug_r[d] for d in unannotated_drugs]
    unannotated_summary = {
        "n_drugs": len(unannotated_drugs),
        "mean_r": round(float(np.mean(unannotated_rs)), 6) if unannotated_rs else None,
        "std_r": round(float(np.std(unannotated_rs)), 6) if unannotated_rs else None,
    }

    # Per-drug flat list
    per_drug_list: list[dict] = []
    for drug, r_val in sorted(pooled_per_drug_r.items()):
        per_drug_list.append({
            "drug": drug,
            "moa": moa.get(drug, "Unknown"),
            "mean_r": round(r_val, 6),
        })

    # ---- Write results ----
    results = {
        "overall_mean_r": round(overall_mean_r, 6),
        "n_drugs": len(pooled_per_drug_r),
        "n_folds": n_folds,
        "per_moa": per_moa,
        "unannotated": unannotated_summary,
        "per_drug": per_drug_list,
    }

    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)
    out_path = report_data / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_path)

    # ---- Summary table ----
    logger.info("=" * 70)
    logger.info("%-30s  %8s  %8s  %5s", "MoA", "mean_r", "std_r", "n")
    logger.info("-" * 70)
    for entry in per_moa:
        flag = ""
        if entry["n_drugs"] < 3:
            flag = " [<3 drugs]"
        logger.info(
            "%-30s  %8.4f  %8.4f  %5d%s",
            entry["moa"][:30], entry["mean_r"], entry["std_r"], entry["n_drugs"], flag,
        )
    logger.info("-" * 70)
    logger.info(
        "%-30s  %8.4f  %8s  %5d",
        "Unannotated", unannotated_summary["mean_r"] or 0.0, "-", unannotated_summary["n_drugs"],
    )
    logger.info(
        "%-30s  %8.4f  %8s  %5d",
        "ALL DRUGS", overall_mean_r, "-", len(pooled_per_drug_r),
    )


if __name__ == "__main__":
    main()
