"""02_lineage_analysis: per-lineage per-drug r and LINCS coverage bias test.

Two reviewer objections tested:
1. LINEAGE CONFOUNDING: is pan-cancer per-drug r inflated by the model learning
   cancer-type → average sensitivity rather than genuine within-lineage ranking?
2. LINCS SELECTION BIAS: are LINCS-covered drugs inherently more predictable
   (regardless of LINCS signature features), making the +0.17 LINCS gain inflated?

Method: Ridge(α=1.0), RNA PCA(550) + mut PCA(200), PASO 10-fold drug-blind CV.
Per-lineage per-drug r restricted to test pairs from each lineage (min_cells=5).
LINCS split: per-drug r on LINCS-covered vs uncovered test drugs.

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

LINEAGE_MAP = {
    "HAEMATOPOIETIC_AND_LYMPHOID_TISSUE": "Hematologic",
    "LUNG": "Lung",
    "SKIN": "Skin",
    "CENTRAL_NERVOUS_SYSTEM": "CNS",
    "BREAST": "Breast",
    "LARGE_INTESTINE": "Colorectal",
}
ALL_LINEAGES = [*sorted(set(LINEAGE_MAP.values())), "Other"]


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
    lincs_drugs: set[str],
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

    rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_rows,
                                     rna_dim=RNA_DIM, mut_dim=MUT_DIM)
    cell_feat = np.concatenate([rna_pca, mut_pca], axis=1)

    tr_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
    te_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
    y_train = train_df["ln_ic50"].values.astype(np.float32)
    y_test = test_df["ln_ic50"].values.astype(np.float32)

    sc = safe_fit_scaler(cell_feat[tr_idx])
    ridge = Ridge(alpha=1.0)
    ridge.fit(sc.transform(cell_feat[tr_idx]), y_train)
    preds = ridge.predict(sc.transform(cell_feat[te_idx])).astype(np.float32)

    d_te = test_df["drug_name"].values
    lin_te = np.array([depmap_to_lineage.get(c, "Other") for c in test_df["depmap_id"]])

    # Pan-cancer
    overall_r = float(mean_per_drug_r(preds, y_test, d_te, min_cells=MIN_CELLS))
    n_drugs = len(np.unique(d_te))
    logger.info("  fold %d overall: per_drug_r=%.4f (%d drugs, %d pairs)",
                fold_i, overall_r, n_drugs, len(test_df))

    # Per-lineage
    lineage_results: dict[str, dict] = {}
    for lin in ALL_LINEAGES:
        mask = lin_te == lin
        if not mask.any():
            lineage_results[lin] = {"mean": float("nan"), "n_drugs": 0, "n_pairs": 0}
            continue
        r = float(mean_per_drug_r(preds[mask], y_test[mask], d_te[mask], min_cells=MIN_CELLS))
        n = len(np.unique(d_te[mask]))
        lineage_results[lin] = {"mean": r, "n_drugs": n, "n_pairs": int(mask.sum())}
        if n > 0:
            logger.info("  fold %d [%-12s]: per_drug_r=%.4f (%d drugs)",
                        fold_i, lin, r, n)

    # LINCS coverage split
    lincs_mask = np.isin(d_te, sorted(lincs_drugs))
    nonlincs_mask = ~lincs_mask
    if lincs_mask.any():
        r_lincs = float(mean_per_drug_r(
            preds[lincs_mask], y_test[lincs_mask], d_te[lincs_mask], min_cells=MIN_CELLS))
        n_lincs = len(np.unique(d_te[lincs_mask]))
    else:
        r_lincs, n_lincs = float("nan"), 0
    if nonlincs_mask.any():
        r_nonlincs = float(mean_per_drug_r(
            preds[nonlincs_mask], y_test[nonlincs_mask], d_te[nonlincs_mask], min_cells=MIN_CELLS))
        n_nonlincs = len(np.unique(d_te[nonlincs_mask]))
    else:
        r_nonlincs, n_nonlincs = float("nan"), 0

    logger.info("  fold %d LINCS covered=%.4f (%d) uncovered=%.4f (%d)",
                fold_i, r_lincs, n_lincs, r_nonlincs, n_nonlincs)

    return {
        "fold": fold_i,
        "overall": {"mean": overall_r, "n_drugs": n_drugs},
        "by_lineage": lineage_results,
        "lincs_covered": {"mean": r_lincs, "n_drugs": n_lincs},
        "lincs_uncovered": {"mean": r_nonlincs, "n_drugs": n_nonlincs},
    }


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
    logger.info("02_lineage_analysis: per-lineage per-drug r and LINCS coverage bias test%s",
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

    with open(DATA_DIR / "lincs_drug_index.json") as f:
        lincs_idx = json.load(f)
    lincs_drugs: set[str] = set(lincs_idx["matched_drugs"])
    logger.info("LINCS matched drugs: %d  unmatched: %d",
                len(lincs_idx["matched_drugs"]), len(lincs_idx["unmatched_drugs"]))

    results_path = report_dir / "results.json"
    all_results: list[dict] = []

    for fold_i in range(k_folds):
        logger.info("=== Fold %d/%d ===", fold_i + 1, k_folds)
        fold_res = run_fold(
            fold_i, rna, mutations, available_cells,
            name_to_depmap, depmap_to_lineage, lincs_drugs,
        )
        if fold_res:
            all_results.append(fold_res)
            results_path.write_text(json.dumps(all_results, indent=2))

    logger.info("=" * 70)
    overall_vals = [r["overall"]["mean"] for r in all_results
                    if not np.isnan(r["overall"]["mean"])]
    logger.info("Overall pan-cancer: %.4f ± %.4f (%d folds)",
                float(np.mean(overall_vals)), float(np.std(overall_vals)), len(overall_vals))

    lineage_summary: dict[str, list[float]] = {}
    for r in all_results:
        for lin, res in r["by_lineage"].items():
            if not np.isnan(res["mean"]) and res["n_drugs"] > 0:
                lineage_summary.setdefault(lin, []).append(res["mean"])
    for lin in ALL_LINEAGES:
        if lin in lineage_summary:
            vals = lineage_summary[lin]
            logger.info("  %-20s %.4f ± %.4f (%d folds)",
                        lin, float(np.mean(vals)), float(np.std(vals)), len(vals))

    lincs_vals = [r["lincs_covered"]["mean"] for r in all_results
                  if not np.isnan(r["lincs_covered"]["mean"])]
    nonlincs_vals = [r["lincs_uncovered"]["mean"] for r in all_results
                     if not np.isnan(r["lincs_uncovered"]["mean"])]
    logger.info("LINCS covered: %.4f ± %.4f  uncovered: %.4f ± %.4f",
                float(np.mean(lincs_vals)) if lincs_vals else float("nan"),
                float(np.std(lincs_vals)) if lincs_vals else float("nan"),
                float(np.mean(nonlincs_vals)) if nonlincs_vals else float("nan"),
                float(np.std(nonlincs_vals)) if nonlincs_vals else float("nan"))
    logger.info("Done. Results: %s", results_path)


if __name__ == "__main__":
    main()
