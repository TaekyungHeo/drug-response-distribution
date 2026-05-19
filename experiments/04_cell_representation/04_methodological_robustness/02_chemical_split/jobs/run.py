"""02_chemical_split: Tanimoto-distance drug-blind CV vs PASO random split.

Tests whether the r=0.631 ceiling is inflated by chemical similarity leakage in
random drug-blind splits. Tanimoto-cluster 10-fold split holds out drugs that are
maximally dissimilar from training drugs.

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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
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


def tanimoto_distance_matrix(fp: np.ndarray) -> np.ndarray:
    """Tanimoto distance (1 - Jaccard similarity) for binary fingerprints."""
    fp = fp.astype(np.float32)
    inter = fp @ fp.T                          # (n, n) intersection counts
    counts = fp.sum(axis=1)                    # (n,) popcount per drug
    union = counts[:, None] + counts[None, :] - inter  # (n, n) union counts
    sim = np.where(union > 0, inter / union, 0.0)
    return (1.0 - sim).astype(np.float64)


def run_tanimoto_fold(
    fold_i: int,
    test_drugs: set[str],
    dr: pd.DataFrame,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
) -> dict:
    """Run Ridge for one chemical-split fold."""
    train_df = dr[~dr["drug_name"].isin(test_drugs)].copy()
    test_df = dr[dr["drug_name"].isin(test_drugs)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return {"per_drug_r": float("nan"), "n_train_drugs": 0, "n_test_drugs": 0}

    train_cells = sorted(train_df["depmap_id"].unique())
    all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
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
    drug_names = test_df["drug_name"].values

    sc = safe_fit_scaler(cell_feat[tr_idx])
    ridge = Ridge(alpha=1.0)
    ridge.fit(sc.transform(cell_feat[tr_idx]), y_train)
    preds = ridge.predict(sc.transform(cell_feat[te_idx])).astype(np.float32)

    r = float(mean_per_drug_r(preds, y_test, drug_names, min_cells=MIN_CELLS))
    n_test_drugs = len(np.unique(drug_names))
    n_train_drugs = len(train_df["drug_name"].unique())
    logger.info("  tanimoto fold %d: per_drug_r=%.4f (%d train, %d test drugs)",
                fold_i, r, n_train_drugs, n_test_drugs)
    return {"per_drug_r": r, "n_train_drugs": n_train_drugs, "n_test_drugs": n_test_drugs}


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
    logger.info("02_chemical_split: Tanimoto drug-blind CV vs PASO random split%s",
                " [SMOKE]" if args.smoke else "")

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    available_cells = set(rna.index) & set(mutations.index)
    name_to_depmap = load_cell_line_index(DATA_DIR)
    logger.info("available cells: %d", len(available_cells))

    # Collect all 233 PASO drugs and build sorted list (matches fingerprint_233 order)
    paso_drugs: set[str] = set()
    for k in range(K_FOLDS):
        tr, te = load_paso_pairs(PASO_FOLDS_DIR, name_to_depmap, available_cells, k)
        paso_drugs |= set(tr["drug_name"]) | set(te["drug_name"])
    paso_drugs_sorted = sorted(paso_drugs)
    logger.info("PASO drugs: %d", len(paso_drugs_sorted))

    # Load fingerprints (233×2048, ordered as sorted PASO drugs)
    fp = np.load(DATA_DIR / "drug_fingerprints_233.npy")
    logger.info("Fingerprints shape: %s", fp.shape)

    # Tanimoto distance matrix + hierarchical clustering → 10 clusters
    dist_mat = tanimoto_distance_matrix(fp)
    np.fill_diagonal(dist_mat, 0.0)
    logger.info("Tanimoto distance: mean=%.4f  max=%.4f", dist_mat.mean(), dist_mat.max())

    Z = linkage(squareform(dist_mat), method="ward")
    labels = fcluster(Z, t=K_FOLDS, criterion="maxclust")  # 1-indexed clusters
    cluster_sizes = [int((labels == k).sum()) for k in range(1, K_FOLDS + 1)]
    logger.info("Chemical cluster sizes: %s", cluster_sizes)

    # Drug response for chemical-split CV (all PASO drugs, available cells)
    dr_raw = pd.read_parquet(DATA_DIR / "drug_response.parquet")
    dr = dr_raw[
        dr_raw["depmap_id"].isin(available_cells) & dr_raw["drug_name"].isin(paso_drugs)
    ].copy().reset_index(drop=True)
    logger.info("Drug response: %d pairs", len(dr))

    # ── PASO random split (reference) ──
    logger.info("=== PASO random split (reference) ===")
    random_folds: list[dict] = []
    for fold_i in range(k_folds):
        tr, te = load_paso_pairs(PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i)
        if len(tr) == 0 or len(te) == 0:
            continue
        all_cells_f = sorted(set(tr["depmap_id"]) | set(te["depmap_id"]))
        train_cells_f = sorted(tr["depmap_id"].unique())
        cell_to_row_f = {c: i for i, c in enumerate(all_cells_f)}
        rna_arr = rna.loc[all_cells_f].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells_f].values.astype(np.float32)
        train_rows_f = np.array([cell_to_row_f[c] for c in train_cells_f], dtype=np.int32)
        rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_rows_f,
                                         rna_dim=RNA_DIM, mut_dim=MUT_DIM)
        cell_feat = np.concatenate([rna_pca, mut_pca], axis=1)
        tr_idx = np.array([cell_to_row_f[c] for c in tr["depmap_id"]], dtype=np.int32)
        te_idx = np.array([cell_to_row_f[c] for c in te["depmap_id"]], dtype=np.int32)
        sc = safe_fit_scaler(cell_feat[tr_idx])
        ridge = Ridge(alpha=1.0)
        ridge.fit(sc.transform(cell_feat[tr_idx]), tr["ln_ic50"].values.astype(np.float32))
        preds = ridge.predict(sc.transform(cell_feat[te_idx])).astype(np.float32)
        r = float(mean_per_drug_r(preds, te["ln_ic50"].values.astype(np.float32),
                                  te["drug_name"].values, min_cells=MIN_CELLS))
        logger.info("  random fold %d: per_drug_r=%.4f", fold_i, r)
        random_folds.append({"per_drug_r": r})

    # ── Tanimoto chemical split ──
    logger.info("=== Tanimoto chemical split ===")
    tanimoto_folds: list[dict] = []
    results_path = report_dir / "results.json"

    for fold_i in range(k_folds):
        cluster_id = fold_i + 1  # clusters are 1-indexed
        test_drug_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        test_drugs = {paso_drugs_sorted[i] for i in test_drug_indices}
        logger.info("=== Chemical fold %d: %d test drugs ===", fold_i + 1, len(test_drugs))
        fold_res = run_tanimoto_fold(
            fold_i, test_drugs, dr, rna, mutations
        )
        tanimoto_folds.append(fold_res)
        results_path.write_text(json.dumps({
            "random_folds": random_folds,
            "tanimoto_folds": tanimoto_folds,
        }, indent=2))

    # Summary
    logger.info("=" * 70)
    random_r_vals = [f["per_drug_r"] for f in random_folds]
    tanimoto_r_vals = [f["per_drug_r"] for f in tanimoto_folds if not np.isnan(f["per_drug_r"])]

    random_mean = float(np.mean(random_r_vals))
    tanimoto_mean = float(np.mean(tanimoto_r_vals)) if tanimoto_r_vals else float("nan")
    delta = tanimoto_mean - random_mean if not np.isnan(tanimoto_mean) else float("nan")

    logger.info("Random split:   %.4f ± %.4f", random_mean, float(np.std(random_r_vals)))
    logger.info("Tanimoto split: %.4f ± %.4f  Δ=%.4f",
                tanimoto_mean, float(np.std(tanimoto_r_vals)) if tanimoto_r_vals else float("nan"),
                delta)

    if not np.isnan(delta):
        if abs(delta) <= 0.05:
            verdict = f"Tanimoto split Δ={delta:.3f} ≤ 0.05 — chemical similarity leakage is negligible."
        else:
            verdict = f"Tanimoto split Δ={delta:.3f} > 0.05 — random split is optimistic; chemical leakage present."
    else:
        verdict = "Could not compute Tanimoto split (insufficient data)."
    logger.info("Verdict: %s", verdict)

    output = {
        "random": {
            "per_drug_r_mean": random_mean,
            "per_drug_r_std": float(np.std(random_r_vals)),
        },
        "tanimoto": {
            "per_drug_r_mean": tanimoto_mean,
            "per_drug_r_std": float(np.std(tanimoto_r_vals)) if tanimoto_r_vals else float("nan"),
            "delta_vs_random": delta,
        },
        "cluster_sizes": cluster_sizes,
        "verdict": verdict,
        "fold_results": {"random": random_folds, "tanimoto": tanimoto_folds},
    }
    results_path.write_text(json.dumps(output, indent=2))
    logger.info("Done. Results: %s", results_path)


if __name__ == "__main__":
    main()
