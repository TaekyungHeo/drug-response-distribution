"""Within-MoA leave-one-drug-out Ridge vs all-drug baseline.

For each MoA class with >=3 drugs, hold out one drug at a time, train Ridge
on remaining same-MoA drugs only, predict held-out drug. Compare per-drug r
against the all-drug baseline from 01_diagnosis/01_moa_performance.

CLI:
  --smoke    restrict to 4 focus MoAs only (fast check)

Output:
  report/data/results.json
  report/data/within_moa_results.csv
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
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell
from src.utils.solutions import group_drugs_by_moa, load_moa_annotations

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"
BASELINE_PATH = (
    ROOT / "experiments" / "05_solutions" / "01_diagnosis"
    / "01_moa_performance" / "report" / "data" / "results.json"
)

K_FOLDS = 10
RIDGE_ALPHA = 1.0
MIN_MOA_DRUGS = 3

FOCUS_MOAS = [
    "ERK MAPK signaling",
    "EGFR signaling",
    "Mitosis",
    "Cell cycle",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _drug_r(preds: np.ndarray, targets: np.ndarray) -> float | None:
    """Pearson r between preds and targets, or None if insufficient."""
    if len(preds) < 5:
        return None
    if np.std(preds) < 1e-8 or np.std(targets) < 1e-8:
        return None
    return float(pearsonr(preds, targets)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Within-MoA LOO Ridge")
    parser.add_argument("--smoke", action="store_true", help="Focus MoAs only")
    args = parser.parse_args()

    logger.info("01_within_moa | ROOT=%s | smoke=%s", ROOT, args.smoke)

    # ---- Load baseline per-drug r ----
    if BASELINE_PATH.exists():
        with BASELINE_PATH.open() as f:
            baseline = json.load(f)
        baseline_per_drug: dict[str, float] = {
            d["drug"]: d["mean_r"] for d in baseline["per_drug"]
        }
        logger.info("Loaded baseline: %d drugs", len(baseline_per_drug))
    else:
        logger.warning("Baseline not found at %s — all_drug_r will be NaN", BASELINE_PATH)
        baseline_per_drug = {}

    # ---- Load omics ----
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    available_cells = set(rna.index) & set(mutations.index)

    # ---- MoA annotations ----
    moa = load_moa_annotations()
    logger.info("MoA annotations: %d drugs", len(moa))

    # ---- Build per-fold data: which drugs are in test for each fold ----
    # Load all folds and map pairs
    fold_data: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for k in range(K_FOLDS):
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, k
        )
        fold_data.append((train_df, test_df))

    # Collect all drugs across folds, find which fold each drug is tested in
    drug_to_test_fold: dict[str, int] = {}
    for k, (_, test_df) in enumerate(fold_data):
        for d in test_df["drug_name"].unique():
            drug_to_test_fold[d] = k

    # ---- Group drugs by MoA, filter >=MIN_MOA_DRUGS ----
    all_drug_names = sorted(drug_to_test_fold.keys())
    moa_groups = group_drugs_by_moa(all_drug_names, moa)
    moa_groups = {
        m: drugs for m, drugs in moa_groups.items()
        if len(drugs) >= MIN_MOA_DRUGS
    }
    if args.smoke:
        moa_groups = {
            m: drugs for m, drugs in moa_groups.items()
            if m in FOCUS_MOAS
        }
    logger.info(
        "MoA groups (>=%d drugs): %d classes, %d drugs total",
        MIN_MOA_DRUGS,
        len(moa_groups),
        sum(len(v) for v in moa_groups.values()),
    )

    # ---- Run within-MoA LOO ----
    # For each fold, pre-compute cell features once, then iterate MoA/drug combos
    within_moa_per_drug: dict[str, float] = {}

    # Group work by fold for efficiency (one PCA fit per fold)
    fold_to_work: dict[int, list[tuple[str, str, list[str]]]] = {}
    for moa_label, drugs in moa_groups.items():
        for held_out in drugs:
            k = drug_to_test_fold.get(held_out)
            if k is None:
                continue
            train_drugs = [d for d in drugs if d != held_out]
            if len(train_drugs) < 2:
                continue
            fold_to_work.setdefault(k, []).append((moa_label, held_out, train_drugs))

    for fold_i in sorted(fold_to_work.keys()):
        t0 = datetime.now()
        work_items = fold_to_work[fold_i]
        logger.info(
            "Fold %d: %d LOO tasks, started at %s",
            fold_i, len(work_items), t0.strftime("%H:%M:%S"),
        )

        train_df, test_df = fold_data[fold_i]
        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty data — skipping", fold_i)
            continue

        # Cell features (PCA fit on train cells only)
        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        cell_to_row = {c: i for i, c in enumerate(all_cells)}
        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        train_cell_rows = np.unique(np.array(
            [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
        ))
        rna_c, mut_c = compress_cell(rna_arr, mut_arr, train_cell_rows)
        cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

        # Pre-index train_df by drug for fast lookup
        train_by_drug: dict[str, pd.DataFrame] = {
            d: grp for d, grp in train_df.groupby("drug_name")
        }
        test_by_drug: dict[str, pd.DataFrame] = {
            d: grp for d, grp in test_df.groupby("drug_name")
        }

        for moa_label, held_out, moa_train_drugs in work_items:
            # Training: same-MoA drugs (excluding held-out) from this fold's train
            train_parts = [
                train_by_drug[d] for d in moa_train_drugs
                if d in train_by_drug
            ]
            if not train_parts:
                continue
            moa_train = pd.concat(train_parts, ignore_index=True)

            # Test: held-out drug from this fold's test
            if held_out not in test_by_drug:
                continue
            moa_test = test_by_drug[held_out]

            if len(moa_train) < 10 or len(moa_test) < 5:
                continue

            # Build feature matrices
            X_train = cell_feat[
                np.array([cell_to_row[c] for c in moa_train["depmap_id"]], dtype=np.int32)
            ]
            y_train = moa_train["ln_ic50"].values.astype(np.float64)
            X_test = cell_feat[
                np.array([cell_to_row[c] for c in moa_test["depmap_id"]], dtype=np.int32)
            ]
            y_test = moa_test["ln_ic50"].values.astype(np.float64)

            # Fit Ridge on same-MoA training data only
            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r = _drug_r(preds, y_test)
            if r is not None:
                within_moa_per_drug[held_out] = r

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info("  Fold %d: completed in %.1fs", fold_i, elapsed)

    # ---- Aggregate results ----
    logger.info("Within-MoA LOO: %d drugs evaluated", len(within_moa_per_drug))

    # Per-drug table
    per_drug_list: list[dict] = []
    for drug, wmoa_r in sorted(within_moa_per_drug.items()):
        all_r = baseline_per_drug.get(drug, float("nan"))
        per_drug_list.append({
            "drug": drug,
            "drug_id": None,
            "moa": moa.get(drug, "Unknown"),
            "all_drug_r": round(all_r, 6),
            "within_moa_r": round(wmoa_r, 6),
        })

    # Per-MoA aggregation
    per_moa_list: list[dict] = []
    for moa_label, drugs in sorted(moa_groups.items()):
        wmoa_rs = [within_moa_per_drug[d] for d in drugs if d in within_moa_per_drug]
        all_rs = [baseline_per_drug.get(d, float("nan")) for d in drugs if d in within_moa_per_drug]
        all_rs_clean = [r for r in all_rs if not np.isnan(r)]
        if not wmoa_rs:
            continue
        wmoa_mean = float(np.mean(wmoa_rs))
        all_mean = float(np.mean(all_rs_clean)) if all_rs_clean else float("nan")
        delta = wmoa_mean - all_mean if not np.isnan(all_mean) else float("nan")
        per_moa_list.append({
            "moa": moa_label,
            "all_drug_mean_r": round(all_mean, 6),
            "within_moa_mean_r": round(wmoa_mean, 6),
            "delta": round(delta, 6),
            "n_drugs": len(wmoa_rs),
            "drugs": sorted([d for d in drugs if d in within_moa_per_drug]),
        })
    per_moa_list.sort(key=lambda x: x["delta"], reverse=True)

    # Overall
    all_wmoa_rs = list(within_moa_per_drug.values())
    all_base_rs = [baseline_per_drug.get(d, float("nan")) for d in within_moa_per_drug]
    all_base_clean = [r for r in all_base_rs if not np.isnan(r)]
    overall = {
        "all_drug_mean_r": round(float(np.mean(all_base_clean)), 6) if all_base_clean else None,
        "within_moa_mean_r": round(float(np.mean(all_wmoa_rs)), 6),
        "n_moa_classes": len(per_moa_list),
        "n_drugs": len(within_moa_per_drug),
    }

    results = {
        "overall": overall,
        "per_moa": per_moa_list,
        "per_drug": per_drug_list,
    }

    # ---- Write outputs ----
    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)

    out_json = report_data / "results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Written: %s", out_json)

    # CSV
    csv_rows = []
    for d in per_drug_list:
        csv_rows.append({
            "drug": d["drug"],
            "moa": d["moa"],
            "all_drug_r": d["all_drug_r"],
            "within_moa_r": d["within_moa_r"],
            "delta": round(d["within_moa_r"] - d["all_drug_r"], 6)
            if not np.isnan(d["all_drug_r"]) else float("nan"),
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = report_data / "within_moa_results.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info("Written: %s", csv_path)

    # ---- Validation: check focus MoA baselines match ----
    if baseline_per_drug:
        logger.info("=" * 70)
        logger.info("Validation: focus MoA baseline check")
        for entry in per_moa_list:
            if entry["moa"] in FOCUS_MOAS:
                logger.info(
                    "  %-25s all_drug=%.4f  within_moa=%.4f  delta=%+.4f  n=%d",
                    entry["moa"],
                    entry["all_drug_mean_r"],
                    entry["within_moa_mean_r"],
                    entry["delta"],
                    entry["n_drugs"],
                )

    # ---- Summary table ----
    logger.info("=" * 70)
    logger.info(
        "%-30s  %8s  %8s  %8s  %5s",
        "MoA", "all_drug", "w/MoA", "delta", "n",
    )
    logger.info("-" * 70)
    for entry in per_moa_list:
        logger.info(
            "%-30s  %8.4f  %8.4f  %+8.4f  %5d",
            entry["moa"][:30],
            entry["all_drug_mean_r"],
            entry["within_moa_mean_r"],
            entry["delta"],
            entry["n_drugs"],
        )
    logger.info("-" * 70)
    logger.info(
        "%-30s  %8s  %8.4f  %8s  %5d",
        "OVERALL",
        f"{overall['all_drug_mean_r']:.4f}" if overall["all_drug_mean_r"] else "N/A",
        overall["within_moa_mean_r"],
        "",
        overall["n_drugs"],
    )


if __name__ == "__main__":
    main()
