"""02_modality_ablations: do additional omics (CNV/metabolomics) help Ridge drug-blind?

Tests whether adding CNV or metabolomics to the canonical RNA+mut Ridge baseline
improves per-drug r on PASO drug-blind CV. Canonical setup throughout.

Conditions (all use Ridge(α=1.0), PASO 10-fold drug-blind CV, per-drug r):
  rna_mut          — RNA PCA(550) + mut PCA(200)   [§canonical baseline = 0.631]
  rna_mut_cnv      — adds CNV PCA(300)
  rna_mut_metab    — adds metabolomics (225 features, no PCA needed)
  rna_mut_all      — RNA+mut+CNV+metabolomics

Usage:
  python run.py                           # all conditions → results.json
  python run.py --condition rna_mut       # single → results_rna_mut.json
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
from src.utils.ridge import compress_multi_omics, safe_fit_scaler  # noqa: E402

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
MIN_CELLS_EVAL = 5

# PCA dims by modality (fit on training cells only)
PCA_DIMS = {"rna": 550, "mutations": 200, "cnv": 300}
# metabolomics (225) is small enough to skip PCA

ABLATION_CONDITIONS: dict[str, list[str]] = {
    "rna_mut":       ["rna", "mutations"],
    "rna_mut_cnv":   ["rna", "mutations", "cnv"],
    "rna_mut_metab": ["rna", "mutations", "metabolomics"],
    "rna_mut_all":   ["rna", "mutations", "cnv", "metabolomics"],
}
ALL_CONDITION_NAMES = list(ABLATION_CONDITIONS.keys())


def run_condition(
    name: str,
    modalities: list[str],
    omics: dict[str, pd.DataFrame],
    available_cells_by_condition: dict[str, set[str]],
    name_to_depmap: dict[str, str],
    k_folds: int,
) -> dict:
    """Run PASO k_folds-fold drug-blind CV for one modality condition."""
    available = available_cells_by_condition[name]
    logger.info("=== Condition: %s | available_cells=%d ===", name, len(available))

    folds_out = []
    for fold_i in range(k_folds):
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available, fold_i
        )
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("  fold %d: empty — skip", fold_i)
            continue

        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        train_cells = sorted(train_df["depmap_id"].unique())
        cell_feat, cell_to_row = compress_multi_omics(
            omics, modalities, all_cells, train_cells, pca_dims=PCA_DIMS
        )

        tr_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
        te_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
        y_train = train_df["ln_ic50"].values.astype(np.float32)
        y_test = test_df["ln_ic50"].values.astype(np.float32)

        sc = safe_fit_scaler(cell_feat[tr_idx])
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(sc.transform(cell_feat[tr_idx]), y_train)
        preds = ridge.predict(sc.transform(cell_feat[te_idx])).astype(np.float32)

        per_dr = mean_per_drug_r(preds, y_test, test_df["drug_name"].values, min_cells=MIN_CELLS_EVAL)
        n_test_drugs = int(test_df["drug_name"].nunique())
        logger.info("  fold %d: n_train=%d n_test=%d n_test_drugs=%d per_drug_r=%.4f feat_dim=%d",
                    fold_i, len(train_df), len(test_df), n_test_drugs, per_dr, cell_feat.shape[1])
        folds_out.append({"per_drug_r": per_dr, "feat_dim": cell_feat.shape[1]})

    per_drug_rs = [f["per_drug_r"] for f in folds_out]
    return {
        "condition": name,
        "modalities": modalities,
        "folds": folds_out,
        "per_drug_r_mean": float(np.mean(per_drug_rs)) if per_drug_rs else float("nan"),
        "per_drug_r_std": float(np.std(per_drug_rs)) if per_drug_rs else float("nan"),
    }


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
    logger.info("02_modality_ablations: Ridge multi-omics addition on PASO drug-blind per-drug r%s%s",
                f" [cond={args.condition}]" if args.condition else "",
                " [SMOKE]" if args.smoke else "")

    if args.condition:
        results_path = report_dir / f"results_{args.condition}.json"
    else:
        results_path = report_dir / "results.json"

    # Load only modalities needed for active conditions
    needed_modalities: set[str] = set()
    for cname in active_conditions:
        needed_modalities.update(ABLATION_CONDITIONS[cname])
    logger.info("Loading omics: %s", sorted(needed_modalities))

    omics: dict[str, pd.DataFrame] = {}
    modality_files = {
        "rna":          DATA_DIR / "rna.parquet",
        "mutations":    DATA_DIR / "mutations.parquet",
        "cnv":          DATA_DIR / "cnv.parquet",
        "metabolomics": DATA_DIR / "metabolomics.parquet",
    }
    for mod in sorted(needed_modalities):
        omics[mod] = pd.read_parquet(modality_files[mod])
        logger.info("  %s: %s", mod, omics[mod].shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)

    # Precompute available cells per active condition
    base_cells = set(omics["rna"].index) & set(omics["mutations"].index)
    available_cells_by_condition: dict[str, set[str]] = {}
    for cname in active_conditions:
        cells = base_cells.copy()
        for mod in ABLATION_CONDITIONS[cname]:
            if mod not in ("rna", "mutations"):
                cells &= set(omics[mod].index)
        available_cells_by_condition[cname] = cells
        logger.info("  %s: %d cells available", cname, len(cells))

    # Run active conditions
    results: dict[str, dict] = {}
    baseline_mean: float | None = None
    for cname in active_conditions:
        res = run_condition(cname, ABLATION_CONDITIONS[cname], omics,
                            available_cells_by_condition=available_cells_by_condition,
                            name_to_depmap=name_to_depmap,
                            k_folds=k_folds)
        results[cname] = res
        if cname == "rna_mut":
            baseline_mean = res["per_drug_r_mean"]

    # Compute deltas vs baseline
    for _cname, res in results.items():
        if baseline_mean is not None:
            res["delta_vs_rna_mut"] = round(res["per_drug_r_mean"] - baseline_mean, 4)

    # Summary
    logger.info("=" * 60)
    logger.info("%-20s  %10s  %6s  %s", "Condition", "per-drug r", "±std", "delta")
    for cname, res in results.items():
        logger.info("%-20s  %10.4f  %6.4f  %+.4f",
                    cname, res["per_drug_r_mean"], res["per_drug_r_std"],
                    res.get("delta_vs_rna_mut", 0.0))

    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
