"""02_active_selection: Which K cells to screen for maximum K-shot gain?

Compares four cell selection strategies for K-shot response matching:
  Random   - uniform random sample of K cells (baseline)
  MaxVar   - K cells with highest inter-drug IC50 variance in training data
  MidResp  - K cells whose mean IC50 is closest to the global median
  Diverse  - farthest-first traversal in RNA PCA(550) space

For each strategy at K in {1,3,5,10,20}: select K cells, run response
matching (response_match_predict), compute per-drug Pearson r on
non-anchor cells.  Cell statistics computed from TRAINING data only.

Uses PASO 10-fold drug-blind CV.

CLI:
  --smoke    2 folds, K={1,5}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell
from src.utils.solutions import response_match_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
K_VALUES = [1, 3, 5, 10, 20]
N_RANDOM_DRAWS = 10  # average over multiple random draws for the Random strategy
BLEND_WEIGHT = 0.5
N_NEIGHBORS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Cell selection strategies
# ---------------------------------------------------------------------------

def select_random(
    n_cells: int, k: int, rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random sample of K cells."""
    return rng.choice(n_cells, size=min(k, n_cells), replace=False)


def select_maxvar(
    response_matrix: np.ndarray, k: int,
) -> np.ndarray:
    """Select K cells with highest inter-drug response variance.

    response_matrix: (n_train_drugs, n_cells). NaN-aware variance per cell.
    """
    n_cells = response_matrix.shape[1]
    variances = np.nanvar(response_matrix, axis=0)
    # Break ties deterministically
    k_sel = min(k, n_cells)
    return np.argsort(variances)[-k_sel:][::-1]


def select_midresp(
    response_matrix: np.ndarray, k: int,
) -> np.ndarray:
    """Select K cells whose mean response is closest to the global median.

    response_matrix: (n_train_drugs, n_cells). NaN-aware.
    """
    n_cells = response_matrix.shape[1]
    cell_means = np.nanmean(response_matrix, axis=0)
    global_median = np.nanmedian(response_matrix)
    dist_to_median = np.abs(cell_means - global_median)
    k_sel = min(k, n_cells)
    return np.argsort(dist_to_median)[:k_sel]


def select_diverse(
    rna_pca: np.ndarray, k: int,
) -> np.ndarray:
    """Farthest-first traversal in RNA PCA space.

    rna_pca: (n_cells, pca_dim).
    Start from the cell closest to the centroid, then greedily pick the
    farthest cell from the selected set.
    """
    n_cells = rna_pca.shape[0]
    k_sel = min(k, n_cells)

    # Start from cell closest to centroid
    centroid = rna_pca.mean(axis=0, keepdims=True)
    dists_to_centroid = cdist(rna_pca, centroid, metric="euclidean").ravel()
    selected = [int(np.argmin(dists_to_centroid))]

    # Pre-compute pairwise distances
    if n_cells <= 2000:
        all_dists = cdist(rna_pca, rna_pca, metric="euclidean")
    else:
        all_dists = None  # compute on the fly for very large panels

    for _ in range(k_sel - 1):
        if all_dists is not None:
            min_dist_to_selected = all_dists[:, selected].min(axis=1)
        else:
            sel_pts = rna_pca[selected]
            min_dist_to_selected = cdist(rna_pca, sel_pts, metric="euclidean").min(axis=1)
        min_dist_to_selected[selected] = -1.0  # exclude already selected
        selected.append(int(np.argmax(min_dist_to_selected)))

    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# Per-drug evaluation on non-anchor cells
# ---------------------------------------------------------------------------

def evaluate_drug(
    train_response_matrix: np.ndarray,
    test_drug_response: np.ndarray,
    anchor_idx: np.ndarray,
    cell_mean: np.ndarray,
) -> float | None:
    """Run response matching for one test drug and return Pearson r on non-anchor cells.

    Returns None if insufficient non-anchor cells or constant predictions.
    """
    n_cells = train_response_matrix.shape[1]
    anchor_set = set(anchor_idx.tolist())
    non_anchor = np.array([j for j in range(n_cells) if j not in anchor_set], dtype=int)

    # Observed values at anchor cells
    observed = test_drug_response[anchor_idx]
    # Skip if all observed are NaN
    if np.all(np.isnan(observed)):
        return None

    pred = response_match_predict(
        train_response_matrix=train_response_matrix,
        test_observed=observed,
        anchor_cell_idx=anchor_idx,
        cell_mean=cell_mean,
        blend_weight=BLEND_WEIGHT,
        n_neighbors=N_NEIGHBORS,
    )

    # Evaluate on non-anchor cells with valid (non-NaN) true values
    true_vals = test_drug_response[non_anchor]
    pred_vals = pred[non_anchor]
    ok = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
    if ok.sum() < 5:
        return None
    t, p = true_vals[ok], pred_vals[ok]
    if t.std() < 1e-8 or p.std() < 1e-8:
        return None
    return float(pearsonr(t, p)[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 folds, K={1,5}")
    args = parser.parse_args()

    t_start = time.perf_counter()
    logger.info(
        "02_active_selection | started at %s | smoke=%s",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        args.smoke,
    )

    n_folds = 2 if args.smoke else K_FOLDS
    k_values = [1, 5] if args.smoke else K_VALUES
    strategies = ["Random", "MaxVar", "MidResp", "Diverse"]

    # ---- Load omics ----
    rna_df = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations_df = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna_df.shape, mutations_df.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    available_cells = set(rna_df.index) & set(mutations_df.index)

    # ---- Collect per-drug results across folds ----
    # Key: (strategy, k, drug) -> list of r values across folds
    results_by_key: dict[tuple[str, int, str], list[float]] = {}

    for fold_i in range(n_folds):
        t_fold = time.perf_counter()
        logger.info("Fold %d/%d", fold_i, n_folds)

        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i,
        )
        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty train or test -- skipping", fold_i)
            continue

        # ---- Build cell universe for this fold ----
        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        cell_to_col = {c: i for i, c in enumerate(all_cells)}
        n_cells = len(all_cells)

        # ---- Cell features: PCA compress for Diverse strategy ----
        rna_arr = rna_df.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations_df.loc[all_cells].values.astype(np.float32)
        train_cell_set = sorted(set(train_df["depmap_id"]))
        train_cell_rows = np.array(
            [cell_to_col[c] for c in train_cell_set], dtype=np.int32,
        )
        rna_c, _ = compress_cell(rna_arr, mut_arr, train_cell_rows)
        # rna_c is shape (n_cells, 550) -- used for Diverse strategy

        # ---- Build train response matrix (n_train_drugs, n_cells) ----
        train_drugs = sorted(train_df["drug_name"].unique())
        train_drug_to_row = {d: i for i, d in enumerate(train_drugs)}
        n_train_drugs = len(train_drugs)

        train_response = np.full((n_train_drugs, n_cells), np.nan, dtype=np.float64)
        for _, row in train_df.iterrows():
            di = train_drug_to_row[row["drug_name"]]
            ci = cell_to_col[row["depmap_id"]]
            train_response[di, ci] = row["ln_ic50"]

        # ---- Cell mean from training data ----
        cell_mean = np.nanmean(train_response, axis=0)
        # Fill remaining NaN with global mean
        global_mean = np.nanmean(train_response)
        cell_mean = np.where(np.isnan(cell_mean), global_mean, cell_mean)

        # ---- Pre-compute deterministic cell selections ----
        maxvar_cache: dict[int, np.ndarray] = {}
        midresp_cache: dict[int, np.ndarray] = {}
        diverse_cache: dict[int, np.ndarray] = {}

        for k in k_values:
            maxvar_cache[k] = select_maxvar(train_response, k)
            midresp_cache[k] = select_midresp(train_response, k)
            diverse_cache[k] = select_diverse(rna_c, k)

        # ---- Sanity checks ----
        if fold_i == 0:
            for k in k_values:
                mv_var = np.nanvar(train_response, axis=0)[maxvar_cache[k]].mean()
                rng_tmp = np.random.default_rng(RANDOM_SEED)
                rand_idx = select_random(n_cells, k, rng_tmp)
                rand_var = np.nanvar(train_response, axis=0)[rand_idx].mean()
                logger.info(
                    "  Sanity K=%d: MaxVar mean_var=%.4f, Random mean_var=%.4f",
                    k, mv_var, rand_var,
                )
                if k > 1:
                    div_pts = rna_c[diverse_cache[k]]
                    div_dists = cdist(div_pts, div_pts, metric="euclidean")
                    rand_pts = rna_c[rand_idx]
                    rand_dists = cdist(rand_pts, rand_pts, metric="euclidean")
                    logger.info(
                        "  Sanity K=%d: Diverse mean_pairwise_dist=%.2f, Random=%.2f",
                        k, div_dists.mean(), rand_dists.mean(),
                    )

        # ---- Build test drug response vectors ----
        test_drugs = sorted(test_df["drug_name"].unique())
        test_drug_responses: dict[str, np.ndarray] = {}
        for drug in test_drugs:
            vec = np.full(n_cells, np.nan, dtype=np.float64)
            drug_rows = test_df[test_df["drug_name"] == drug]
            for _, row in drug_rows.iterrows():
                ci = cell_to_col[row["depmap_id"]]
                vec[ci] = row["ln_ic50"]
            test_drug_responses[drug] = vec

        # ---- Evaluate each strategy x K ----
        for strategy in strategies:
            for k in k_values:
                if strategy == "Random":
                    # Average over multiple random draws
                    for drug in test_drugs:
                        rs_draws: list[float] = []
                        for draw in range(N_RANDOM_DRAWS):
                            rng = np.random.default_rng(RANDOM_SEED + fold_i * 1000 + draw)
                            anchor_idx = select_random(n_cells, k, rng)
                            r_val = evaluate_drug(
                                train_response, test_drug_responses[drug],
                                anchor_idx, cell_mean,
                            )
                            if r_val is not None:
                                rs_draws.append(r_val)
                        if rs_draws:
                            key = ("Random", k, drug)
                            results_by_key.setdefault(key, []).append(float(np.mean(rs_draws)))
                else:
                    if strategy == "MaxVar":
                        anchor_idx = maxvar_cache[k]
                    elif strategy == "MidResp":
                        anchor_idx = midresp_cache[k]
                    elif strategy == "Diverse":
                        anchor_idx = diverse_cache[k]
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")

                    for drug in test_drugs:
                        r_val = evaluate_drug(
                            train_response, test_drug_responses[drug],
                            anchor_idx, cell_mean,
                        )
                        if r_val is not None:
                            key = (strategy, k, drug)
                            results_by_key.setdefault(key, []).append(r_val)

        elapsed_fold = time.perf_counter() - t_fold
        logger.info("  Fold %d: %d test drugs, elapsed=%.1fs", fold_i, len(test_drugs), elapsed_fold)

    # ---- Aggregate results ----
    # Per-drug: average across folds
    per_drug_rows: list[dict] = []
    for (strategy, k, drug), rs in results_by_key.items():
        per_drug_rows.append({
            "drug": drug,
            "strategy": strategy,
            "k": k,
            "mean_r": float(np.mean(rs)),
            "n_folds": len(rs),
        })

    # Strategy x K summary
    strategy_k_agg: dict[tuple[str, int], list[float]] = {}
    for row in per_drug_rows:
        key = (row["strategy"], row["k"])
        strategy_k_agg.setdefault(key, []).append(row["mean_r"])

    # Random baseline for delta computation
    random_baseline: dict[int, float] = {}
    for k in k_values:
        rk = ("Random", k)
        if rk in strategy_k_agg:
            random_baseline[k] = float(np.mean(strategy_k_agg[rk]))

    strategy_by_k: list[dict] = []
    for (strategy, k), rs in sorted(strategy_k_agg.items()):
        mean_r = float(np.mean(rs))
        std_r = float(np.std(rs))
        delta = mean_r - random_baseline.get(k, mean_r)
        strategy_by_k.append({
            "strategy": strategy,
            "k": k,
            "mean_r": round(mean_r, 6),
            "std_r": round(std_r, 6),
            "delta_vs_random": round(delta, 6),
            "n_drugs": len(rs),
        })

    # ---- Write outputs ----
    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "overall": {
            "random_baseline_r_by_k": {
                str(k): round(random_baseline.get(k, float("nan")), 6)
                for k in k_values
            },
        },
        "strategy_by_k": strategy_by_k,
        "per_drug": per_drug_rows,
        "n_folds": n_folds,
        "k_values": k_values,
        "strategies": strategies,
        "smoke": args.smoke,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
    }

    results_path = report_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", results_path)

    # strategy_comparison.csv
    csv_path = report_dir / "strategy_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["strategy", "k", "mean_r", "std_r", "delta_vs_random", "n_drugs"],
        )
        writer.writeheader()
        for row in strategy_by_k:
            writer.writerow(row)
    logger.info("Strategy comparison written to %s", csv_path)

    # per_drug_by_strategy.csv
    csv_path2 = report_dir / "per_drug_by_strategy.csv"
    with csv_path2.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["drug", "strategy", "k", "mean_r", "n_folds"],
        )
        writer.writeheader()
        for row in sorted(per_drug_rows, key=lambda r: (r["strategy"], r["k"], r["drug"])):
            writer.writerow(row)
    logger.info("Per-drug results written to %s", csv_path2)

    # ---- Summary table ----
    logger.info("=" * 70)
    logger.info("STRATEGY x K SUMMARY")
    logger.info("=" * 70)
    logger.info("%-10s  %5s  %8s  %8s  %10s  %5s", "Strategy", "K", "mean_r", "std_r", "delta_rand", "n")
    logger.info("-" * 70)
    for row in sorted(strategy_by_k, key=lambda r: (r["k"], r["strategy"])):
        logger.info(
            "%-10s  %5d  %8.4f  %8.4f  %+10.4f  %5d",
            row["strategy"], row["k"], row["mean_r"], row["std_r"],
            row["delta_vs_random"], row["n_drugs"],
        )
    logger.info("=" * 70)
    logger.info("Elapsed: %.1f s", time.perf_counter() - t_start)
    logger.info("Done.")


if __name__ == "__main__":
    main()
