"""04_foundation_model: does scFoundation (768-dim) break the per-drug r ceiling?

Conditions (all Ridge(α=1.0), PASO 10-fold drug-blind CV on scFoundation-covered cells):
  A_rna_mut_pca : RNA PCA(550) + mut PCA(200)  [baseline restricted to scFoundation cells]
  B_scfoundation: scFoundation 768-dim embeddings
  C_concat      : RNA PCA(550) + mut PCA(200) + scFoundation concat

Usage:
  python run.py                              # all conditions → results.json
  python run.py --condition A_rna_mut_pca   # single → results_A_rna_mut_pca.json
  python run.py --smoke                      # 1 fold only

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
SCF_DIR = ROOT / "data" / "external" / "scFoundation"
EXP_DIR = Path(__file__).parents[1]

K_FOLDS = 10
MIN_CELLS = 5
RNA_DIM, MUT_DIM = 550, 200

ALL_CONDITIONS = ["A_rna_mut_pca", "B_scfoundation", "C_concat"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 fold only")
    parser.add_argument("--condition", type=str, choices=ALL_CONDITIONS, default=None,
                        help="Run a single condition (writes results_<condition>.json)")
    args = parser.parse_args()
    k_folds = 1 if args.smoke else K_FOLDS
    active_conditions = [args.condition] if args.condition else ALL_CONDITIONS

    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(logs_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("04_foundation_model: scFoundation vs RNA-PCA per-drug r%s%s",
                f" [cond={args.condition}]" if args.condition else "",
                " [SMOKE]" if args.smoke else "")

    if args.condition:
        results_path = report_dir / f"results_{args.condition}.json"
    else:
        results_path = report_dir / "results.json"

    # Load scFoundation embeddings
    scf_emb = np.load(SCF_DIR / "50M-0.1B-res_embedding.npy")  # (561, 768)
    scf_cells: list[str] = []
    with open(SCF_DIR / "cancer_cell_line.info") as f:
        for line in f:
            parts = line.strip().split("\t")
            scf_cells.append(parts[0])
    assert len(scf_cells) == scf_emb.shape[0], (
        f"scFoundation cell count mismatch: {len(scf_cells)} names vs {scf_emb.shape[0]} embeddings"
    )
    scf_cell_to_row = {c: i for i, c in enumerate(scf_cells)}
    logger.info("scFoundation: %d cells, embedding dim=%d", len(scf_cells), scf_emb.shape[1])

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    available_cells = set(rna.index) & set(mutations.index) & set(scf_cells)
    logger.info("RNA: %s  mut: %s  scFoundation cells: %d  available: %d",
                rna.shape, mutations.shape, len(scf_cells), len(available_cells))

    name_to_depmap = load_cell_line_index(DATA_DIR)

    fold_results: dict[str, list[dict]] = {c: [] for c in active_conditions}
    pred_corr_ab: list[float] = []

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
        scf_arr = np.array([scf_emb[scf_cell_to_row[c]] for c in all_cells], dtype=np.float32)

        train_rows = np.array([cell_to_row[c] for c in train_cells], dtype=np.int32)
        tr_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
        te_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
        y_train = train_df["ln_ic50"].values.astype(np.float32)
        y_test = test_df["ln_ic50"].values.astype(np.float32)
        d_te = test_df["drug_name"].values

        logger.info("  train=%d pairs  test=%d pairs  train_cells=%d",
                    len(train_df), len(test_df), len(train_cells))

        needs_pca = any(c in active_conditions for c in ["A_rna_mut_pca", "C_concat"])
        rna_pca = mut_pca = None
        if needs_pca:
            rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_rows,
                                             rna_dim=RNA_DIM, mut_dim=MUT_DIM)

        preds_fold: dict[str, np.ndarray] = {}
        for cname in active_conditions:
            if cname == "A_rna_mut_pca":
                assert rna_pca is not None and mut_pca is not None
                Xtr = np.c_[rna_pca[tr_idx], mut_pca[tr_idx]]
                Xte = np.c_[rna_pca[te_idx], mut_pca[te_idx]]
            elif cname == "B_scfoundation":
                Xtr = scf_arr[tr_idx]
                Xte = scf_arr[te_idx]
            elif cname == "C_concat":
                assert rna_pca is not None and mut_pca is not None
                Xtr = np.c_[rna_pca[tr_idx], mut_pca[tr_idx], scf_arr[tr_idx]]
                Xte = np.c_[rna_pca[te_idx], mut_pca[te_idx], scf_arr[te_idx]]
            else:
                continue

            sc = safe_fit_scaler(Xtr)
            ridge = Ridge(alpha=1.0)
            ridge.fit(sc.transform(Xtr), y_train)
            preds = ridge.predict(sc.transform(Xte)).astype(np.float32)
            preds_fold[cname] = preds

            per_dr = float(mean_per_drug_r(preds, y_test, d_te, min_cells=MIN_CELLS))
            fold_results[cname].append({"per_drug_r": per_dr})

        # Pred-corr between A and B (only when both conditions active)
        if "A_rna_mut_pca" in preds_fold and "B_scfoundation" in preds_fold:
            corr_ab = float(pearsonr(preds_fold["A_rna_mut_pca"],
                                     preds_fold["B_scfoundation"]).statistic)
            pred_corr_ab.append(corr_ab)
            logger.info("  fold %d: " + " | ".join(
                f"{c[:12]}: {fold_results[c][-1]['per_drug_r']:.4f}" for c in active_conditions
            ) + f"  Pearson(A,B)={corr_ab:.4f}", fold_i)
        else:
            logger.info("  fold %d: " + " | ".join(
                f"{c[:12]}: {fold_results[c][-1]['per_drug_r']:.4f}" for c in active_conditions
            ), fold_i)

        results_path.write_text(json.dumps({
            "condition": args.condition,
            "fold_results": fold_results,
            "pred_corr_ab": pred_corr_ab,
        }, indent=2))

    logger.info("=" * 60)
    base_vals = fold_results.get("A_rna_mut_pca", [])
    base_mean = float(np.mean([f["per_drug_r"] for f in base_vals])) if base_vals else float("nan")

    summary: dict[str, dict] = {}
    for cname in active_conditions:
        vals = [f["per_drug_r"] for f in fold_results[cname]]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals))
        delta = m - base_mean
        summary[cname] = {
            "per_drug_r_mean": round(m, 4),
            "per_drug_r_std": round(s, 4),
            "delta_vs_A": round(delta, 4) if not np.isnan(base_mean) else None,
        }
        logger.info("  %-20s  per-drug r=%.4f ± %.4f", cname, m, s)

    mean_corr = float(np.mean(pred_corr_ab)) if pred_corr_ab else float("nan")
    if not np.isnan(mean_corr):
        logger.info("Mean Pearson(preds_A, preds_B) = %.4f  (reference: 1.0000)", mean_corr)

    out: dict = {
        "condition": args.condition,
        "summary": summary,
        "fold_results": fold_results,
        "pred_corr_ab": [round(r, 4) for r in pred_corr_ab],
        "mean_pred_corr_ab": round(mean_corr, 4) if not np.isnan(mean_corr) else None,
    }

    if len(active_conditions) > 1 and not np.isnan(base_mean) and "B_scfoundation" in summary:
        d = summary["B_scfoundation"]["delta_vs_A"] or 0.0
        if abs(d) < 0.005:
            verdict = "CONFIRMED: scFoundation converges to same per-drug r as RNA-PCA — ceiling is fundamental"
        elif d > 0.01:
            verdict = "SURPRISING: scFoundation improves per-drug r — novel biological signal"
        else:
            verdict = f"MARGINAL: Δ={d:+.4f}"
        logger.info("VERDICT: %s", verdict)
        out["verdict"] = verdict

    results_path.write_text(json.dumps(out, indent=2))
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
