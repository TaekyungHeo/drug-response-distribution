"""03_representation_sweep: do richer cell encodings break the drug-blind ceiling?

Conditions (all Ridge, PASO 10-fold drug-blind CV, per-drug r, no drug features):
  baseline     : RNA PCA(550) + mut PCA(200), α=1.0     [canonical]
  pca_1500     : RNA PCA(1500) + mut PCA(200), α=10.0
  pca_max      : RNA PCA(n_train-1) + mut PCA(n_train-1), α=10.0
  full_rna     : raw RNA (19k) + raw mut (no PCA), α=100.0
  pathway_kegg : KEGG pathway scores (1284 features), α=1.0
  rna_plus_path: RNA PCA(550) + pathway features, α=1.0

Usage:
  python run.py                           # all conditions → results.json
  python run.py --condition baseline      # single → results_baseline.json
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

K_FOLDS = 10
MIN_CELLS = 5

ALL_CONDITIONS: list[tuple[str, float]] = [
    ("baseline",      1.0),
    ("pca_1500",      10.0),
    ("pca_max",       10.0),
    ("full_rna",      100.0),
    ("pathway_kegg",  1.0),
    ("rna_plus_path", 1.0),
]
ALL_CONDITION_NAMES = [n for n, _ in ALL_CONDITIONS]
ALPHA_BY_NAME = {n: a for n, a in ALL_CONDITIONS}


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
    logger.info("03_representation_sweep: 6 cell encoding alternatives vs Ridge ceiling%s%s",
                f" [cond={args.condition}]" if args.condition else "",
                " [SMOKE]" if args.smoke else "")

    if args.condition:
        results_path = report_dir / f"results_{args.condition}.json"
    else:
        results_path = report_dir / "results.json"

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    # pathway needed only if pathway_kegg or rna_plus_path active
    needs_pathway = any(c in active_conditions for c in ["pathway_kegg", "rna_plus_path"])
    if needs_pathway:
        pathway = pd.read_parquet(DATA_DIR / "pathway_features.parquet")
        available_cells = set(rna.index) & set(mutations.index) & set(pathway.index)
        logger.info("RNA: %s  mut: %s  pathway: %s  available: %d cells",
                    rna.shape, mutations.shape, pathway.shape, len(available_cells))
    else:
        pathway = None
        available_cells = set(rna.index) & set(mutations.index)
        logger.info("RNA: %s  mut: %s  available: %d cells",
                    rna.shape, mutations.shape, len(available_cells))

    name_to_depmap = load_cell_line_index(DATA_DIR)

    fold_results: dict[str, list[dict]] = {c: [] for c in active_conditions}

    for fold_i in range(k_folds):
        logger.info("=== Fold %d/%d ===", fold_i + 1, k_folds)
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
        )
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("fold %d: empty split — skip", fold_i)
            continue

        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        train_cells = sorted(train_df["depmap_id"].unique())
        cell_to_row = {c: i for i, c in enumerate(all_cells)}

        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        pw_arr = pathway.loc[all_cells].values.astype(np.float32) if pathway is not None else None
        train_rows = np.array([cell_to_row[c] for c in train_cells], dtype=np.int32)

        tr_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
        te_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
        y_train = train_df["ln_ic50"].values.astype(np.float32)
        y_test = test_df["ln_ic50"].values.astype(np.float32)
        d_te = test_df["drug_name"].values

        logger.info("  train=%d pairs  test=%d pairs  train_cells=%d",
                    len(train_df), len(test_df), len(train_cells))

        # Precompute only what's needed for active conditions
        needs_rna550 = any(c in active_conditions for c in ["baseline", "rna_plus_path"])
        needs_mut200_base = any(c in active_conditions for c in ["baseline"])
        needs_pca_1500 = "pca_1500" in active_conditions
        needs_pca_max = "pca_max" in active_conditions

        rna_550 = mut_200 = rna_1500 = rna_max = mut_max = None

        if needs_rna550 or needs_mut200_base or needs_pca_1500:
            rna_550, mut_200 = compress_cell(rna_arr, mut_arr, train_rows,
                                             rna_dim=550, mut_dim=200)
        if needs_pca_1500:
            rna_1500, _mut_tmp = compress_cell(rna_arr, mut_arr, train_rows,
                                               rna_dim=1500, mut_dim=200)
            if mut_200 is None:
                mut_200 = _mut_tmp
        if needs_pca_max:
            rna_max, mut_max = compress_cell(
                rna_arr, mut_arr, train_rows,
                rna_dim=rna_arr.shape[1], mut_dim=mut_arr.shape[1],
            )

        for cname in active_conditions:
            alpha = ALPHA_BY_NAME[cname]
            if cname == "baseline":
                assert rna_550 is not None and mut_200 is not None
                Xtr = np.c_[rna_550[tr_idx], mut_200[tr_idx]]
                Xte = np.c_[rna_550[te_idx], mut_200[te_idx]]
            elif cname == "pca_1500":
                assert rna_1500 is not None and mut_200 is not None
                Xtr = np.c_[rna_1500[tr_idx], mut_200[tr_idx]]
                Xte = np.c_[rna_1500[te_idx], mut_200[te_idx]]
            elif cname == "pca_max":
                assert rna_max is not None and mut_max is not None
                Xtr = np.c_[rna_max[tr_idx], mut_max[tr_idx]]
                Xte = np.c_[rna_max[te_idx], mut_max[te_idx]]
            elif cname == "full_rna":
                Xtr = np.c_[rna_arr[tr_idx], mut_arr[tr_idx]]
                Xte = np.c_[rna_arr[te_idx], mut_arr[te_idx]]
            elif cname == "pathway_kegg":
                assert pw_arr is not None
                Xtr = pw_arr[tr_idx]
                Xte = pw_arr[te_idx]
            elif cname == "rna_plus_path":
                assert rna_550 is not None and pw_arr is not None
                Xtr = np.c_[rna_550[tr_idx], pw_arr[tr_idx]]
                Xte = np.c_[rna_550[te_idx], pw_arr[te_idx]]
            else:
                continue

            sc = safe_fit_scaler(Xtr)
            ridge = Ridge(alpha=alpha)
            ridge.fit(sc.transform(Xtr), y_train)
            preds = ridge.predict(sc.transform(Xte)).astype(np.float32)

            per_dr = float(mean_per_drug_r(preds, y_test, d_te, min_cells=MIN_CELLS))
            global_r = float(pearsonr(preds, y_test).statistic)
            fold_results[cname].append({"per_drug_r": per_dr, "global_r": global_r})

        fold_line = " | ".join(
            f"{c[:10]}: {fold_results[c][-1]['per_drug_r']:.4f}"
            for c in active_conditions if fold_results[c]
        )
        logger.info("  fold %d: %s", fold_i, fold_line)

        results_path.write_text(json.dumps({
            "condition": args.condition,
            "fold_results": fold_results,
        }, indent=2))

    logger.info("=" * 60)
    base_vals = fold_results.get("baseline", [])
    base_mean = float(np.mean([f["per_drug_r"] for f in base_vals])) if base_vals else float("nan")

    summary: dict[str, dict] = {}
    for cname in active_conditions:
        vals = [f["per_drug_r"] for f in fold_results[cname]]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals))
        gm = float(np.mean([f["global_r"] for f in fold_results[cname]]))
        delta = m - base_mean
        summary[cname] = {
            "alpha": ALPHA_BY_NAME[cname],
            "per_drug_r_mean": round(m, 4),
            "per_drug_r_std": round(s, 4),
            "global_r_mean": round(gm, 4),
            "delta_vs_baseline": round(delta, 4) if not np.isnan(base_mean) else None,
        }
        logger.info("  %-18s  per-drug r=%.4f ± %.4f  global r=%.4f  Δ=%+.4f",
                    cname, m, s, gm, delta)

    out = {
        "condition": args.condition,
        "summary": summary,
        "fold_results": fold_results,
    }
    results_path.write_text(json.dumps(out, indent=2))
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
