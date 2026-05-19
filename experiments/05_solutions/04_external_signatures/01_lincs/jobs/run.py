"""LINCS L1000 signatures as drug features in Ridge regression.

Tests whether LINCS L1000 consensus transcriptional signatures improve
global r and/or per-drug r beyond the cell-only baseline. Both conditions
are evaluated on the SAME ~104-drug subset (drugs with LINCS coverage).

Conditions:
  no_drug        — cell features only, 104-drug subset
  lincs          — cell features + LINCS PCA(64), 104-drug subset
  random_vector  — cell features + random N(0,1) 64-dim, 104-drug subset

Output: EXP_DIR/report/data/results.json, lincs_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import pearson_r
from src.evaluation.per_drug import per_drug_r
from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell, normalize_continuous_fold

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
RIDGE_ALPHA = 1.0
LINCS_PCA_DIM = 64
RNG_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LINCS loading and PCA
# ---------------------------------------------------------------------------

def load_lincs_signatures(
    data_dir: Path,
    lincs_drugs: set[str],
) -> tuple[np.ndarray, list[str]]:
    """Load raw LINCS signatures for drugs in lincs_drugs set.

    Returns (signatures, drug_names) where signatures[i] corresponds to
    drug_names[i].
    """
    sig_path = data_dir / "lincs_signatures.npy"
    idx_path = data_dir / "lincs_drug_index.json"

    raw = np.load(sig_path).astype(np.float32)
    with idx_path.open() as f:
        idx_data = json.load(f)
    matched = idx_data["matched_drugs"][: raw.shape[0]]

    keep = [(i, d) for i, d in enumerate(matched) if d in lincs_drugs]
    rows = [i for i, _ in keep]
    names = [d for _, d in keep]
    return raw[rows], names


def fit_lincs_pca(
    signatures: np.ndarray,
    n_components: int = LINCS_PCA_DIM,
) -> tuple[np.ndarray, float]:
    """PCA-reduce LINCS signatures. Returns (reduced, variance_explained)."""
    n_comp = min(n_components, signatures.shape[0] - 1, signatures.shape[1])
    pca = PCA(n_components=n_comp, random_state=RNG_SEED)
    reduced = pca.fit_transform(signatures.astype(np.float64)).astype(np.float32)
    var_explained = float(pca.explained_variance_ratio_.sum())
    return reduced, var_explained


# ---------------------------------------------------------------------------
# Single-fold Ridge
# ---------------------------------------------------------------------------

def run_fold(
    fold_i: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    drug_feats: dict[str, np.ndarray] | None,
    drug_to_idx: dict[str, int],
    condition: str,
) -> dict | None:
    """Run one fold for one condition. Returns metrics dict or None."""
    # Cell features
    all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
    cell_to_row = {c: i for i, c in enumerate(all_cells)}
    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    train_cell_rows = np.array(
        [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
    )
    rna_c, mut_c = compress_cell(rna_arr, mut_arr, np.unique(train_cell_rows))
    cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

    # Drug feature normalization (z-score fit on train drugs)
    train_drugs = sorted(train_df["drug_name"].unique())
    train_drug_idxs = np.array(
        [drug_to_idx[d] for d in train_drugs], dtype=np.int32
    )

    drug_feat_norm: np.ndarray | None = None
    if drug_feats is not None:
        drug_feat_norm = normalize_continuous_fold(drug_feats, train_drug_idxs)

    # Build pair matrices
    def make_X(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows_c = np.array(
            [cell_to_row[c] for c in df["depmap_id"]], dtype=np.int32
        )
        rows_d = np.array(
            [drug_to_idx[d] for d in df["drug_name"]], dtype=np.int32
        )
        y = df["ln_ic50"].values.astype(np.float32)
        Xc = cell_feat[rows_c]
        if drug_feat_norm is not None:
            Xd = drug_feat_norm[rows_d]
            X = np.concatenate([Xc, Xd], axis=1)
        else:
            X = Xc
        return X, y

    X_train, y_train = make_X(train_df)
    X_test, y_test = make_X(test_df)
    drug_names_test = test_df["drug_name"].values

    logger.info(
        "  fold %d | %s: train=%d test=%d n_features=%d",
        fold_i, condition, len(y_train), len(y_test), X_train.shape[1],
    )

    # Fit Ridge
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
    preds = model.predict(X_test.astype(np.float64)).astype(np.float32)

    # Metrics
    global_r = pearson_r(y_test, preds)
    pdr = per_drug_r(preds, y_test, drug_names_test, min_cells=5)
    mean_r = float(np.mean(list(pdr.values()))) if pdr else float("nan")

    logger.info(
        "  fold %d | %s: global_r=%.4f per_drug_r=%.4f n_drugs=%d",
        fold_i, condition, global_r, mean_r, len(pdr),
    )

    return {
        "global_r": global_r,
        "mean_per_drug_r": mean_r,
        "per_drug_r": pdr,
        "n_drugs": len(pdr),
        "fold": fold_i,
        "preds": preds,
        "targets": y_test,
        "drug_names": drug_names_test,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LINCS L1000 drug features")
    parser.add_argument("--smoke", action="store_true", help="Quick run: 2 folds")
    args = parser.parse_args()

    n_folds = 2 if args.smoke else K_FOLDS

    log_dir = EXP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    )
    logging.getLogger().addHandler(fh)

    logger.info("01_lincs | ROOT=%s | folds=%d", ROOT, n_folds)

    # --- Load omics ---
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    available_cells = set(rna.index) & set(mutations.index)

    # --- Build PASO drug set ---
    all_paso_drugs: set[str] = set()
    for k in range(K_FOLDS):
        tr = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{k}.csv")
        te = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{k}.csv")
        all_paso_drugs |= set(tr["drug"].unique()) | set(te["drug"].unique())
    logger.info("PASO drug set: %d drugs", len(all_paso_drugs))

    # --- Load LINCS and identify overlap ---
    with (DATA_DIR / "lincs_drug_index.json").open() as f:
        lincs_index = json.load(f)
    all_lincs_drugs = set(lincs_index["matched_drugs"])
    overlap_drugs = sorted(all_paso_drugs & all_lincs_drugs)
    n_overlap = len(overlap_drugs)
    logger.info(
        "Drug overlap: %d PASO x %d LINCS-matched = %d overlap",
        len(all_paso_drugs), len(all_lincs_drugs), n_overlap,
    )

    # Build drug index restricted to overlap drugs
    drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(sorted(all_paso_drugs))}

    # --- LINCS PCA ---
    lincs_raw, lincs_names = load_lincs_signatures(DATA_DIR, set(overlap_drugs))
    lincs_pca, var_explained = fit_lincs_pca(lincs_raw, LINCS_PCA_DIM)
    logger.info(
        "LINCS PCA(%d): %d drugs, variance explained = %.4f",
        LINCS_PCA_DIM, len(lincs_names), var_explained,
    )

    # Build full drug feature matrix (n_paso_drugs, 64) with zeros for non-LINCS drugs
    n_drugs = len(drug_to_idx)
    lincs_feat = np.zeros((n_drugs, lincs_pca.shape[1]), dtype=np.float32)
    for i, drug in enumerate(lincs_names):
        if drug in drug_to_idx:
            lincs_feat[drug_to_idx[drug]] = lincs_pca[i]

    # Random control: same shape, fixed seed
    rng = np.random.default_rng(RNG_SEED)
    random_feat = rng.standard_normal((n_drugs, lincs_pca.shape[1])).astype(np.float32)

    # --- Conditions ---
    conditions: dict[str, np.ndarray | None] = {
        "no_drug": None,
        "lincs": lincs_feat,
        "random_vector": random_feat,
    }

    # --- Run folds ---
    overlap_set = set(overlap_drugs)
    # Accumulators: condition -> {pooled per-drug r, global preds/targets}
    results: dict[str, dict] = {}

    for condition, drug_feats in conditions.items():
        logger.info("=== Condition: %s ===", condition)
        pooled_pdr: dict[str, float] = {}
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_drug_names: list[np.ndarray] = []

        for fold_i in range(n_folds):
            train_df, test_df = load_paso_pairs(
                PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
            )
            # Filter BOTH train and test to overlap drugs only
            train_df = pd.DataFrame(
                train_df[train_df["drug_name"].isin(overlap_set)]
            )
            test_df = pd.DataFrame(
                test_df[test_df["drug_name"].isin(overlap_set)]
            )
            if train_df.empty or test_df.empty:
                logger.warning("fold %d | %s: empty after overlap filter", fold_i, condition)
                continue

            res = run_fold(
                fold_i=fold_i,
                train_df=train_df,
                test_df=test_df,
                rna=rna,
                mutations=mutations,
                drug_feats=drug_feats,
                drug_to_idx=drug_to_idx,
                condition=condition,
            )
            if res is not None:
                pooled_pdr.update(res["per_drug_r"])
                all_preds.append(res["preds"])
                all_targets.append(res["targets"])
                all_drug_names.append(res["drug_names"])

        # Aggregate
        if all_preds:
            pooled_preds = np.concatenate(all_preds)
            pooled_targets = np.concatenate(all_targets)
            global_r = pearson_r(pooled_targets, pooled_preds)
        else:
            global_r = float("nan")

        mean_pdr = float(np.mean(list(pooled_pdr.values()))) if pooled_pdr else float("nan")
        results[condition] = {
            "global_r": global_r,
            "per_drug_r": mean_pdr,
            "n_drugs": len(pooled_pdr),
            "pooled_per_drug": pooled_pdr,
        }
        logger.info(
            "%s: global_r=%.4f per_drug_r=%.4f n_drugs=%d",
            condition, global_r, mean_pdr, len(pooled_pdr),
        )

    # --- Per-drug delta table ---
    no_drug_pdr = results["no_drug"]["pooled_per_drug"]
    lincs_pdr = results["lincs"]["pooled_per_drug"]
    per_drug_table: list[dict] = []
    for drug in sorted(set(no_drug_pdr) & set(lincs_pdr)):
        per_drug_table.append({
            "drug": drug,
            "no_drug_r": round(no_drug_pdr[drug], 6),
            "lincs_r": round(lincs_pdr[drug], 6),
            "delta": round(lincs_pdr[drug] - no_drug_pdr[drug], 6),
        })

    # --- Build output ---
    output = {
        "drug_overlap": {
            "n_gdsc2": len(all_paso_drugs),
            "n_lincs": len(all_lincs_drugs),
            "n_overlap": n_overlap,
            "overlap_drugs": overlap_drugs,
        },
        "lincs_pca": {
            "n_components": LINCS_PCA_DIM,
            "variance_explained": round(var_explained, 4),
        },
        "comparison": {
            cond: {
                "global_r": round(r["global_r"], 6),
                "per_drug_r": round(r["per_drug_r"], 6),
                "n_drugs": r["n_drugs"],
            }
            for cond, r in results.items()
        },
        "per_drug": per_drug_table,
    }

    # --- Write outputs ---
    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)

    results_path = report_data / "results.json"
    with results_path.open("w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", results_path)

    # Flat CSV
    csv_path = report_data / "lincs_comparison.csv"
    if per_drug_table:
        pd.DataFrame(per_drug_table).to_csv(csv_path, index=False)
        logger.info("CSV written to %s", csv_path)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("%-15s  %8s  %10s  %6s", "Condition", "global_r", "per_drug_r", "n_drugs")
    for cond, r in results.items():
        logger.info(
            "%-15s  %8.4f  %10.4f  %6d",
            cond, r["global_r"], r["per_drug_r"], r["n_drugs"],
        )


if __name__ == "__main__":
    main()
