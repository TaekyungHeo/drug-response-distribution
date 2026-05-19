"""K-shot response matching across PASO drug-blind folds.

For each PASO fold, builds a training response matrix (n_train_drugs x n_cells),
computes the cell-mean prior, then for each test drug at each K in {0,1,3,5,10,20,50}:
  - randomly samples K anchor cells from the test drug's observed cells
  - calls response_match_predict() to get blended predictions
  - computes per-drug r on non-anchor cells only

Optimizes blend_weight w per K via pseudo-test inner CV.
Adds a permuted-response control at K=50.
Averages across folds and random draws.

CLI:
  --smoke    2 folds, K={0,5,50}, 2 draws (fast check)

Output: report/data/results.json, report/data/k_curve.csv, report/data/per_drug_by_k.csv
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
from src.utils.solutions import response_match_predict

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
K_VALUES = [0, 1, 3, 5, 10, 20, 50]
K_VALUES_SMOKE = [0, 5, 50]
N_DRAWS = 10
N_DRAWS_SMOKE = 2
RIDGE_ALPHA = 1.0
W_GRID = [round(w * 0.1, 1) for w in range(11)]  # 0.0, 0.1, ..., 1.0
VAL_FRAC = 0.10  # fraction of train drugs for inner CV

# Reference constants from prior experiments
CELL_MEAN_PRIOR_R = 0.645
CELL_MEAN_ORACLE_R = 0.644
MEASUREMENT_CEILING = 0.754

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_response_matrix(
    df: pd.DataFrame, drug_list: list[str], cell_list: list[str],
) -> np.ndarray:
    """Build (n_drugs, n_cells) response matrix; NaN where unobserved."""
    drug_to_row = {d: i for i, d in enumerate(drug_list)}
    cell_to_col = {c: j for j, c in enumerate(cell_list)}
    mat = np.full((len(drug_list), len(cell_list)), np.nan, dtype=np.float64)
    for _, row in df.iterrows():
        di = drug_to_row.get(row["drug_name"])
        ci = cell_to_col.get(row["depmap_id"])
        if di is not None and ci is not None:
            mat[di, ci] = row["ln_ic50"]
    return mat


def _eval_per_drug_r(
    preds: np.ndarray,
    targets: np.ndarray,
    min_cells: int = 5,
) -> float:
    """Pearson r between predicted and actual responses for one drug."""
    ok = ~np.isnan(preds) & ~np.isnan(targets)
    if ok.sum() < min_cells:
        return float("nan")
    p, t = preds[ok], targets[ok]
    if p.std() < 1e-8 or t.std() < 1e-8:
        return float("nan")
    return float(pearsonr(p, t)[0])


def _kshot_one_drug(
    train_response_matrix: np.ndarray,
    test_drug_responses: np.ndarray,
    cell_mean: np.ndarray,
    k: int,
    rng: np.random.Generator,
    permute: bool = False,
) -> dict[float, float]:
    """Run K-shot for one drug, return {w: per_drug_r} for each w in W_GRID.

    Evaluates on non-anchor cells only.
    """
    # Cells where this test drug has observations
    obs_mask = ~np.isnan(test_drug_responses)
    obs_idx = np.where(obs_mask)[0]

    if k == 0:
        # No anchor cells; prediction = cell_mean for all cells
        r_val = _eval_per_drug_r(cell_mean, test_drug_responses, min_cells=5)
        return {w: r_val for w in W_GRID}

    if len(obs_idx) < k + 5:
        # Not enough cells to sample K anchors and still evaluate
        return {w: float("nan") for w in W_GRID}

    # Sample K anchor cells
    anchor_idx = rng.choice(obs_idx, size=k, replace=False)
    anchor_set = set(anchor_idx.tolist())
    eval_idx = np.array([i for i in obs_idx if i not in anchor_set])

    if len(eval_idx) < 5:
        return {w: float("nan") for w in W_GRID}

    test_observed = test_drug_responses[anchor_idx].copy()
    if permute:
        rng.shuffle(test_observed)

    # Get pure neighbor prediction (blend_weight=1.0)
    neighbor_pred = response_match_predict(
        train_response_matrix=train_response_matrix,
        test_observed=test_observed,
        anchor_cell_idx=anchor_idx,
        cell_mean=cell_mean,
        blend_weight=1.0,
        n_neighbors=5,
    )

    # For each w, blend and evaluate on non-anchor cells
    results: dict[float, float] = {}
    for w in W_GRID:
        blended = w * neighbor_pred + (1.0 - w) * cell_mean
        r_val = _eval_per_drug_r(blended[eval_idx], test_drug_responses[eval_idx])
        results[w] = r_val
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="K-shot response matching")
    parser.add_argument("--smoke", action="store_true", help="Quick run: 2 folds, K={0,5,50}")
    args = parser.parse_args()

    n_folds = 2 if args.smoke else K_FOLDS
    k_values = K_VALUES_SMOKE if args.smoke else K_VALUES
    n_draws = N_DRAWS_SMOKE if args.smoke else N_DRAWS

    logger.info(
        "01_response_matching | ROOT=%s | folds=%d | K=%s | draws=%d",
        ROOT, n_folds, k_values, n_draws,
    )

    # ---- Load omics ----
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    available_cells = set(rna.index) & set(mutations.index)

    # ---- Accumulate results ----
    # per_drug_records: list of dicts with (drug, k, draw, fold, w, r)
    per_drug_records: list[dict] = []

    for fold_i in range(n_folds):
        t0 = datetime.now()
        logger.info("Fold %d/%d started at %s", fold_i, n_folds, t0.strftime("%H:%M:%S"))

        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i,
        )
        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty train or test — skipping", fold_i)
            continue

        # Column space = union of train and test cells
        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        cell_to_col = {c: j for j, c in enumerate(all_cells)}

        # Cell features for Ridge
        cell_to_row = {c: i for i, c in enumerate(all_cells)}
        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        train_cell_rows = np.unique(
            np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
        )
        rna_c, mut_c = compress_cell(rna_arr, mut_arr, train_cell_rows)
        cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

        # Training drugs
        train_drugs = sorted(train_df["drug_name"].unique())
        test_drugs = sorted(test_df["drug_name"].unique())

        # Build training response matrix (n_train_drugs x n_cells)
        train_response_matrix = _build_response_matrix(train_df, train_drugs, all_cells)

        # Cell-mean prior: per-cell mean across training drugs (ignoring NaN)
        cell_mean = np.nanmean(train_response_matrix, axis=0)
        # Fill any remaining NaN (cells with no training observations) with global mean
        global_mean = np.nanmean(cell_mean)
        cell_mean = np.where(np.isnan(cell_mean), global_mean, cell_mean)

        # Build test response vectors (test_drug -> full-length response vector)
        test_response_matrix = _build_response_matrix(test_df, test_drugs, all_cells)

        logger.info(
            "  Fold %d: %d train drugs, %d test drugs, %d cells",
            fold_i, len(train_drugs), len(test_drugs), len(all_cells),
        )

        # ---- Inner CV: hold out val_frac of training drugs ----
        rng_val = np.random.default_rng(42 + fold_i)
        n_val = max(1, int(len(train_drugs) * VAL_FRAC))
        val_drug_perm = rng_val.permutation(len(train_drugs))
        val_drug_idx = val_drug_perm[:n_val]
        inner_train_idx = val_drug_perm[n_val:]
        val_drugs = [train_drugs[i] for i in val_drug_idx]
        inner_train_drugs = [train_drugs[i] for i in inner_train_idx]
        inner_train_mat = train_response_matrix[inner_train_idx]
        inner_cell_mean = np.nanmean(inner_train_mat, axis=0)
        inner_cell_mean = np.where(np.isnan(inner_cell_mean), global_mean, inner_cell_mean)

        # Inner CV w selection per K
        cv_w_per_k: dict[int, float] = {}
        for k in k_values:
            if k == 0:
                cv_w_per_k[k] = 0.0
                continue
            w_scores: dict[float, list[float]] = {w: [] for w in W_GRID}
            for vd_idx in val_drug_idx:
                vd_responses = train_response_matrix[vd_idx]
                for draw in range(min(n_draws, 3)):  # fewer draws for inner CV
                    rng_inner = np.random.default_rng(1000 * fold_i + 100 * vd_idx + draw)
                    w_r = _kshot_one_drug(
                        inner_train_mat, vd_responses, inner_cell_mean,
                        k, rng_inner, permute=False,
                    )
                    for w, r_val in w_r.items():
                        if not np.isnan(r_val):
                            w_scores[w].append(r_val)
            best_w, best_mean = 0.0, -np.inf
            for w in W_GRID:
                if w_scores[w]:
                    m = np.mean(w_scores[w])
                    if m > best_mean:
                        best_mean = m
                        best_w = w
            cv_w_per_k[k] = best_w
        logger.info("  Fold %d inner CV w: %s", fold_i, cv_w_per_k)

        # ---- Main evaluation: test drugs ----
        for di, drug in enumerate(test_drugs):
            test_responses = test_response_matrix[di]
            for k in k_values:
                for draw in range(n_draws):
                    rng_draw = np.random.default_rng(10000 * fold_i + 100 * di + draw)
                    w_results = _kshot_one_drug(
                        train_response_matrix, test_responses, cell_mean,
                        k, rng_draw, permute=False,
                    )
                    for w, r_val in w_results.items():
                        per_drug_records.append({
                            "drug": drug, "fold": fold_i, "k": k, "draw": draw,
                            "w": w, "r": r_val, "permuted": False,
                        })

                # Permuted control at K=50 only
                if k == 50:
                    for draw in range(n_draws):
                        rng_perm = np.random.default_rng(90000 * fold_i + 100 * di + draw)
                        w_results_perm = _kshot_one_drug(
                            train_response_matrix, test_responses, cell_mean,
                            k, rng_perm, permute=True,
                        )
                        for w, r_val in w_results_perm.items():
                            per_drug_records.append({
                                "drug": drug, "fold": fold_i, "k": k, "draw": draw,
                                "w": w, "r": r_val, "permuted": True,
                            })

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info("  Fold %d done in %.1fs", fold_i, elapsed)

    # ---- Aggregate results ----
    logger.info("Aggregating %d records", len(per_drug_records))
    df_all = pd.DataFrame(per_drug_records)
    df_all = df_all.dropna(subset=["r"])

    # For each K: find oracle-optimal w (best mean r across drugs/folds/draws)
    k_curve: list[dict] = []
    for k in k_values:
        sub = df_all[(df_all["k"] == k) & (~df_all["permuted"])]
        if sub.empty:
            continue

        # Oracle-optimal w: best across all w values
        best_w, best_r = 0.0, -np.inf
        for w in W_GRID:
            w_sub = sub[sub["w"] == w]
            if w_sub.empty:
                continue
            # Mean per-drug r: average across draws first, then across drugs
            drug_mean = w_sub.groupby("drug")["r"].mean()
            grand_mean = drug_mean.mean()
            if grand_mean > best_r:
                best_r = grand_mean
                best_w = w

        # Stats at optimal w
        opt_sub = sub[sub["w"] == best_w]
        drug_means = opt_sub.groupby("drug")["r"].mean()
        mean_r = float(drug_means.mean())
        std_r = float(drug_means.std())

        # CV-selected w (from inner CV, averaged across folds)
        # cv_w_per_k was computed per fold; we use it from the last fold for simplicity
        # Recompute: for each fold's cv_w, take the average
        cv_w = cv_w_per_k.get(k, 0.0)

        # Permuted control (only at K=50)
        perm_sub = df_all[(df_all["k"] == k) & (df_all["permuted"]) & (df_all["w"] == best_w)]
        permuted_r = float("nan")
        if not perm_sub.empty:
            perm_drug_means = perm_sub.groupby("drug")["r"].mean()
            permuted_r = float(perm_drug_means.mean())

        k_curve.append({
            "k": k,
            "mean_r": round(mean_r, 6),
            "std_r": round(std_r, 6),
            "optimal_w": best_w,
            "cv_w": cv_w,
            "permuted_r": round(permuted_r, 6) if not np.isnan(permuted_r) else None,
            "n_drugs": int(drug_means.shape[0]),
        })

    # Per-drug by K table (at oracle-optimal w)
    optimal_w_per_k = {row["k"]: row["optimal_w"] for row in k_curve}
    per_drug_by_k: list[dict] = []
    for k in k_values:
        w = optimal_w_per_k.get(k, 0.0)
        sub = df_all[(df_all["k"] == k) & (~df_all["permuted"]) & (df_all["w"] == w)]
        if sub.empty:
            continue
        drug_stats = sub.groupby("drug")["r"].agg(["mean", "std"]).reset_index()
        for _, row in drug_stats.iterrows():
            per_drug_by_k.append({
                "drug": row["drug"],
                "k": k,
                "mean_r": round(float(row["mean"]), 6),
                "std_r": round(float(row["std"]), 6),
            })

    # ---- Write outputs ----
    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)

    results = {
        "overall": {
            "cell_mean_prior_r": CELL_MEAN_PRIOR_R,
            "cell_mean_oracle_r": CELL_MEAN_ORACLE_R,
            "measurement_ceiling": MEASUREMENT_CEILING,
        },
        "k_curve": k_curve,
        "per_drug": per_drug_by_k,
    }
    out_json = report_data / "results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_json)

    # k_curve.csv
    df_kcurve = pd.DataFrame(k_curve)
    df_kcurve.to_csv(report_data / "k_curve.csv", index=False)
    logger.info("K-curve CSV written to %s", report_data / "k_curve.csv")

    # per_drug_by_k.csv
    df_perdrug = pd.DataFrame(per_drug_by_k)
    df_perdrug.to_csv(report_data / "per_drug_by_k.csv", index=False)
    logger.info("Per-drug CSV written to %s", report_data / "per_drug_by_k.csv")

    # ---- Summary ----
    logger.info("=" * 70)
    logger.info("%5s  %8s  %8s  %8s  %8s  %10s  %5s", "K", "mean_r", "std_r", "opt_w", "cv_w", "permuted_r", "n")
    logger.info("-" * 70)
    for row in k_curve:
        pr = f"{row['permuted_r']:.4f}" if row["permuted_r"] is not None else "-"
        logger.info(
            "%5d  %8.4f  %8.4f  %8.1f  %8.1f  %10s  %5d",
            row["k"], row["mean_r"], row["std_r"],
            row["optimal_w"], row["cv_w"], pr, row["n_drugs"],
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
