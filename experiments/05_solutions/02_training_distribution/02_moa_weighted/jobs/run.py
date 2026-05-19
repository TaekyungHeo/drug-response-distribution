"""MoA-weighted Ridge training — weight sweep over same-MoA sample upweighting.

For each weight W in {1, 2, 5, 10, 20}, assign sample_weight = W to training
samples from drugs sharing the same MoA as the test drug, and 1 to all others.
Fit Ridge(alpha=1.0) with RNA PCA(550) + mutation PCA(200) cell features under
PASO 10-fold drug-blind CV. Report per-drug Pearson r grouped by MoA.

W=1 is the all-drug baseline and must match 01_diagnosis/01_moa_performance exactly.

CLI:
  --smoke    run only 2 folds and weights {1, 5}

Output:
  report/data/results.json
  report/data/weight_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.evaluation.per_drug import per_drug_r
from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell
from src.utils.solutions import (
    fit_weighted_ridge,
    group_drugs_by_moa,
    load_moa_annotations,
)

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
RIDGE_ALPHA = 1.0
WEIGHTS_FULL = [1, 2, 5, 10, 20]
WEIGHTS_SMOKE = [1, 5]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MoA-weighted Ridge training sweep")
    parser.add_argument("--smoke", action="store_true", help="Run 2 folds, weights {1,5}")
    args = parser.parse_args()

    n_folds = 2 if args.smoke else K_FOLDS
    weights = WEIGHTS_SMOKE if args.smoke else WEIGHTS_FULL
    logger.info(
        "02_moa_weighted | ROOT=%s | folds=%d | weights=%s", ROOT, n_folds, weights
    )

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

    # ---- Run folds ----
    # pooled_per_drug_r[W][drug_name] = r
    pooled: dict[int, dict[str, float]] = {w: {} for w in weights}

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

        # Cell features (shared across all W for this fold)
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

        train_rows_c = np.array(
            [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
        )
        test_rows_c = np.array(
            [cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32
        )
        X_train = cell_feat[train_rows_c]
        y_train = train_df["ln_ic50"].values.astype(np.float64)
        X_test = cell_feat[test_rows_c]
        y_test = test_df["ln_ic50"].values.astype(np.float64)
        drug_names_test = test_df["drug_name"].values
        drug_names_train = train_df["drug_name"].values

        logger.info(
            "  Fold %d: train=%d test=%d n_features=%d",
            fold_i, len(y_train), len(y_test), X_train.shape[1],
        )

        # Build MoA lookup for training drugs
        train_drug_moa: dict[str, str] = {}
        for d in np.unique(drug_names_train):
            if d in moa:
                train_drug_moa[d] = moa[d]

        # Identify distinct MoA classes among test drugs
        test_drugs = list(np.unique(drug_names_test))
        test_moa_classes: dict[str, list[str]] = {}  # moa_label -> [test_drugs]
        unannotated_test: list[str] = []
        for d in test_drugs:
            if d in moa:
                test_moa_classes.setdefault(moa[d], []).append(d)
            else:
                unannotated_test.append(d)

        for w in weights:
            tw0 = datetime.now()

            if w == 1:
                # W=1: no weighting — pass None for exact baseline match
                model = fit_weighted_ridge(
                    X_train.astype(np.float64), y_train, sample_weights=None,
                    alpha=RIDGE_ALPHA,
                )
                preds = model.predict(X_test.astype(np.float64))
                fold_drug_r = per_drug_r(preds, y_test, drug_names_test, min_cells=5)
                pooled[w].update(fold_drug_r)
            else:
                # For each MoA class, fit once and predict for all test drugs
                # in that class.
                base_weights = np.ones(len(y_train), dtype=np.float64)

                # Cache: moa_label -> fitted model predictions on test set
                # We only need to refit when MoA class changes (different
                # training samples get upweighted).
                for moa_label, moa_test_drugs in test_moa_classes.items():
                    # Build sample weights: W for same-MoA training drugs
                    sw = base_weights.copy()
                    same_moa_mask = np.array(
                        [train_drug_moa.get(d) == moa_label for d in drug_names_train],
                        dtype=bool,
                    )
                    sw[same_moa_mask] = float(w)

                    model = fit_weighted_ridge(
                        X_train.astype(np.float64), y_train,
                        sample_weights=sw, alpha=RIDGE_ALPHA,
                    )
                    preds_all = model.predict(X_test.astype(np.float64))

                    # Evaluate only test drugs in this MoA class
                    for td in moa_test_drugs:
                        mask = drug_names_test == td
                        if mask.sum() < 5:
                            continue
                        p, t = preds_all[mask], y_test[mask]
                        if t.std() < 1e-8 or p.std() < 1e-8:
                            continue
                        from scipy.stats import pearsonr
                        pooled[w][td] = float(pearsonr(p, t)[0])

                # Unannotated test drugs: no same-MoA upweighting possible,
                # equivalent to W=1. Reuse W=1 results.
                for td in unannotated_test:
                    if td in pooled[1]:
                        pooled[w][td] = pooled[1][td]

            elapsed = (datetime.now() - tw0).total_seconds()
            n_eval = sum(1 for d in test_drugs if d in pooled[w])
            vals = [pooled[w][d] for d in test_drugs if d in pooled[w]]
            logger.info(
                "  Fold %d W=%2d: %d drugs, mean_r=%.4f, %.1fs",
                fold_i, w, n_eval,
                float(np.mean(vals)) if vals else float("nan"),
                elapsed,
            )

        elapsed_fold = (datetime.now() - t0).total_seconds()
        logger.info("  Fold %d total: %.1fs", fold_i, elapsed_fold)

    # ---- Validation: W=1 must match baseline ----
    baseline_path = (
        ROOT / "experiments" / "05_solutions" / "01_diagnosis"
        / "01_moa_performance" / "report" / "data" / "results.json"
    )
    if baseline_path.exists() and not args.smoke:
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_per_drug = {e["drug"]: e["mean_r"] for e in baseline["per_drug"]}
        diffs = []
        for drug, r_val in pooled[1].items():
            if drug in baseline_per_drug:
                diffs.append(abs(r_val - baseline_per_drug[drug]))
        if diffs:
            max_diff = max(diffs)
            logger.info(
                "Validation: W=1 vs baseline max |diff| = %.2e (%d drugs compared)",
                max_diff, len(diffs),
            )
            if max_diff > 1e-4:
                logger.warning(
                    "W=1 does NOT match baseline within 1e-4! max diff = %.6f",
                    max_diff,
                )
        else:
            logger.warning("Validation: no overlapping drugs to compare with baseline")
    else:
        logger.info("Validation: baseline not found or smoke mode — skipping comparison")

    # ---- Aggregate results ----
    all_drug_names = sorted(
        set().union(*(pooled[w].keys() for w in weights))
    )
    moa_groups = group_drugs_by_moa(all_drug_names, moa)

    # Overall per weight
    overall: list[dict] = []
    for w in weights:
        rs = list(pooled[w].values())
        overall.append({
            "weight": w,
            "mean_per_drug_r": round(float(np.mean(rs)), 6) if rs else None,
            "n_drugs": len(rs),
        })

    # Per-MoA per weight
    per_moa_records: list[dict] = []
    for w in weights:
        for pathway, drugs in sorted(moa_groups.items()):
            rs = [pooled[w][d] for d in drugs if d in pooled[w]]
            if not rs:
                continue
            per_moa_records.append({
                "moa": pathway,
                "weight": w,
                "mean_r": round(float(np.mean(rs)), 6),
                "std_r": round(float(np.std(rs)), 6),
                "n_drugs": len(rs),
            })

    results = {
        "weights": weights,
        "overall": overall,
        "per_moa": per_moa_records,
    }

    # ---- Write results.json ----
    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)
    out_json = report_data / "results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_json)

    # ---- Write weight_sweep.csv ----
    out_csv = report_data / "weight_sweep.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["moa", "weight", "mean_r", "std_r", "n_drugs"])
        writer.writeheader()
        for rec in per_moa_records:
            writer.writerow(rec)
        # Add overall rows
        for o in overall:
            writer.writerow({
                "moa": "ALL",
                "weight": o["weight"],
                "mean_r": o["mean_per_drug_r"],
                "std_r": "",
                "n_drugs": o["n_drugs"],
            })
    logger.info("CSV written to %s", out_csv)

    # ---- Summary ----
    logger.info("=" * 70)
    header = "%-30s" + "".join(f"  W={w:>2}" for w in weights)
    logger.info(header, "MoA")
    logger.info("-" * 70)

    # Build pivot: moa -> {w: mean_r}
    pivot: dict[str, dict[int, float]] = {}
    for rec in per_moa_records:
        pivot.setdefault(rec["moa"], {})[rec["weight"]] = rec["mean_r"]

    for pathway in sorted(pivot.keys()):
        vals = "".join(f"  {pivot[pathway].get(w, float('nan')):5.3f}" for w in weights)
        logger.info("%-30s%s", pathway[:30], vals)

    logger.info("-" * 70)
    overall_vals = "".join(
        f"  {o['mean_per_drug_r']:5.3f}" if o["mean_per_drug_r"] else "    N/A"
        for o in overall
    )
    logger.info("%-30s%s", "ALL DRUGS", overall_vals)


if __name__ == "__main__":
    main()
