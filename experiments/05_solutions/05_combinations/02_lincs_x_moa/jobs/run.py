"""LINCS x within-MoA 2x2 factorial: do LINCS and within-MoA combine?

Four conditions on the SAME evaluation drug subset (LINCS-covered drugs
within focus MoA classes):
  1. all_drug_no_lincs  — all-drug training, cell features only
  2. all_drug_lincs     — all-drug training, cell + LINCS features
  3. within_moa_no_lincs — within-MoA LOO, cell features only
  4. within_moa_lincs   — within-MoA LOO, cell + LINCS features

Training pool:
  - all-drug conditions: all LINCS-covered PASO drugs (matches 04_external_signatures/01_lincs).
  - within-MoA conditions: same-MoA drugs that ALSO have LINCS coverage
    (excludes non-LINCS MoA drugs for symmetry across conditions).

NOTE: within-MoA training is restricted to LINCS-covered same-MoA drugs in
ALL four conditions (even no-LINCS), so per-drug r will differ slightly from
02_training_distribution/01_within_moa which trains on ALL same-MoA drugs.
This is intentional: all conditions must share the same drug subset.

Evaluation: only drugs in focus MoA classes with LINCS coverage.
PASO 10-fold drug-blind CV for all-drug; leave-one-drug-out for within-MoA.

Output: EXP_DIR/report/data/results.json, factorial_results.csv
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import pearson_r
from src.evaluation.per_drug import per_drug_r
from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell, normalize_continuous_fold
from src.utils.solutions import group_drugs_by_moa, load_moa_annotations

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
RIDGE_ALPHA = 1.0
LINCS_PCA_DIM = 64
RNG_SEED = 42
MIN_MOA_DRUGS = 3

FOCUS_MOAS = [
    "ERK MAPK signaling",
    "EGFR signaling",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LINCS loading and PCA (from 04_external_signatures/01_lincs)
# ---------------------------------------------------------------------------

def load_lincs_signatures(
    data_dir: Path,
    lincs_drugs: set[str],
) -> tuple[np.ndarray, list[str]]:
    """Load raw LINCS signatures for drugs in lincs_drugs set."""
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
# Helpers
# ---------------------------------------------------------------------------

def _drug_r(preds: np.ndarray, targets: np.ndarray) -> float | None:
    """Pearson r, or None if insufficient data."""
    if len(preds) < 5:
        return None
    if np.std(preds) < 1e-8 or np.std(targets) < 1e-8:
        return None
    return float(pearsonr(preds, targets)[0])


# ---------------------------------------------------------------------------
# All-drug condition runner (PASO CV)
# ---------------------------------------------------------------------------

def run_all_drug_folds(
    n_folds: int,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    name_to_depmap: dict,
    available_cells: set,
    overlap_drugs: set[str],
    focus_drugs: set[str],
    drug_to_idx: dict[str, int],
    lincs_feat: np.ndarray | None,
    condition: str,
) -> dict:
    """Run PASO drug-blind CV, train on all overlap drugs, eval on focus drugs.

    Returns dict with per_drug_detail, pooled arrays for per-MoA global_r.
    """
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_drug_names: list[np.ndarray] = []

    for fold_i in range(n_folds):
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
        )
        # Train on all overlap drugs; test on overlap drugs only
        train_df = pd.DataFrame(train_df[train_df["drug_name"].isin(overlap_drugs)])
        test_df = pd.DataFrame(test_df[test_df["drug_name"].isin(overlap_drugs)])
        if train_df.empty or test_df.empty:
            continue

        # Cell features
        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        cell_to_row = {c: i for i, c in enumerate(all_cells)}
        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        train_cell_rows = np.unique(np.array(
            [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
        ))
        rna_c, mut_c = compress_cell(rna_arr, mut_arr, train_cell_rows)
        cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

        # Drug feature normalization (fit on overlap training drugs only)
        train_drugs = sorted(train_df["drug_name"].unique())
        train_drug_idxs = np.array(
            [drug_to_idx[d] for d in train_drugs], dtype=np.int32
        )
        drug_feat_norm = None
        if lincs_feat is not None:
            drug_feat_norm = normalize_continuous_fold(lincs_feat, train_drug_idxs)

        # Build pair matrices
        def make_X(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            return X, y, df["drug_name"].values

        X_train, y_train, _ = make_X(train_df)
        X_test, y_test, dn_test = make_X(test_df)

        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
        preds = model.predict(X_test.astype(np.float64)).astype(np.float32)

        # Keep only focus drugs for evaluation
        focus_mask = np.isin(dn_test, list(focus_drugs))
        if focus_mask.sum() == 0:
            continue
        all_preds.append(preds[focus_mask])
        all_targets.append(y_test[focus_mask])
        all_drug_names.append(dn_test[focus_mask])

        logger.info(
            "  fold %d | %s: train=%d test_focus=%d/%d",
            fold_i, condition, len(y_train), focus_mask.sum(), len(y_test),
        )

    # Aggregate
    if not all_preds:
        return {"global_r": float("nan"), "per_drug_r": float("nan"),
                "per_drug_detail": {}, "n_focus_drugs": 0,
                "pooled_preds": np.array([]),
                "pooled_targets": np.array([]),
                "pooled_drug_names": np.array([])}

    pooled_p = np.concatenate(all_preds)
    pooled_t = np.concatenate(all_targets)
    pooled_dn = np.concatenate(all_drug_names)
    global_r = pearson_r(pooled_t, pooled_p)
    pdr = per_drug_r(pooled_p, pooled_t, pooled_dn, min_cells=5)
    mean_pdr = float(np.mean(list(pdr.values()))) if pdr else float("nan")

    logger.info(
        "%s: global_r=%.4f per_drug_r=%.4f n_drugs=%d",
        condition, global_r, mean_pdr, len(pdr),
    )
    return {
        "global_r": global_r,
        "per_drug_r": mean_pdr,
        "per_drug_detail": pdr,
        "n_focus_drugs": len(pdr),
        "pooled_preds": pooled_p,
        "pooled_targets": pooled_t,
        "pooled_drug_names": pooled_dn,
    }


# ---------------------------------------------------------------------------
# Within-MoA condition runner (LOO)
# ---------------------------------------------------------------------------

def run_within_moa(
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    name_to_depmap: dict,
    available_cells: set,
    moa_lincs_groups: dict[str, list[str]],
    drug_to_test_fold: dict[str, int],
    fold_data: list[tuple[pd.DataFrame, pd.DataFrame]],
    drug_to_idx: dict[str, int],
    lincs_feat: np.ndarray | None,
    overlap_set: set[str],
    condition: str,
) -> dict:
    """Within-MoA LOO across focus MoAs. Returns aggregated results."""
    per_drug_results: dict[str, float] = {}
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_drug_names: list[np.ndarray] = []

    # Group work by fold for efficiency
    fold_to_work: dict[int, list[tuple[str, str, list[str]]]] = {}
    for moa_label, drugs in moa_lincs_groups.items():
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
        train_df, test_df = fold_data[fold_i]
        if train_df.empty or test_df.empty:
            continue

        # Cell features (PCA fit on train cells)
        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        cell_to_row = {c: i for i, c in enumerate(all_cells)}
        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        train_cell_rows = np.unique(np.array(
            [cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32
        ))
        rna_c, mut_c = compress_cell(rna_arr, mut_arr, train_cell_rows)
        cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

        # Drug features: z-score fit on LINCS-covered training drugs only
        # (non-LINCS drugs have zero rows — including them distorts variance)
        drug_feat_norm = None
        if lincs_feat is not None:
            train_lincs_drugs = sorted(
                set(train_df["drug_name"].unique()) & overlap_set
            )
            train_drug_idxs = np.array(
                [drug_to_idx[d] for d in train_lincs_drugs],
                dtype=np.int32,
            )
            drug_feat_norm = normalize_continuous_fold(lincs_feat, train_drug_idxs)

        # Index train/test by drug
        train_by_drug = {d: grp for d, grp in train_df.groupby("drug_name")}
        test_by_drug = {d: grp for d, grp in test_df.groupby("drug_name")}

        for moa_label, held_out, moa_train_drugs in work_items:
            # Training: same-MoA drugs (excl held-out) from this fold's train set
            train_parts = [
                train_by_drug[d] for d in moa_train_drugs if d in train_by_drug
            ]
            if not train_parts:
                continue
            moa_train = pd.concat(train_parts, ignore_index=True)

            # Test: held-out drug from this fold's test set
            if held_out not in test_by_drug:
                continue
            moa_test = test_by_drug[held_out]

            if len(moa_train) < 10 or len(moa_test) < 5:
                continue

            # Build feature matrices
            def _make_X(df: pd.DataFrame) -> np.ndarray:
                rows_c = np.array(
                    [cell_to_row[c] for c in df["depmap_id"]], dtype=np.int32
                )
                Xc = cell_feat[rows_c]
                if drug_feat_norm is not None:
                    rows_d = np.array(
                        [drug_to_idx[d] for d in df["drug_name"]], dtype=np.int32
                    )
                    Xd = drug_feat_norm[rows_d]
                    return np.concatenate([Xc, Xd], axis=1)
                return Xc

            X_train = _make_X(moa_train)
            y_train = moa_train["ln_ic50"].values.astype(np.float64)
            X_test = _make_X(moa_test)
            y_test = moa_test["ln_ic50"].values.astype(np.float64)

            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r = _drug_r(preds, y_test)
            if r is not None:
                per_drug_results[held_out] = r
                all_preds.append(preds.astype(np.float32))
                all_targets.append(y_test.astype(np.float32))
                all_drug_names.append(np.array([held_out] * len(y_test)))

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info("  fold %d | %s: %d tasks, %.1fs", fold_i, condition, len(work_items), elapsed)

    # Aggregate
    if not all_preds:
        return {"global_r": float("nan"), "per_drug_r": float("nan"),
                "per_drug_detail": {}, "n_focus_drugs": 0,
                "pooled_preds": np.array([]),
                "pooled_targets": np.array([]),
                "pooled_drug_names": np.array([])}

    pooled_p = np.concatenate(all_preds)
    pooled_t = np.concatenate(all_targets)
    pooled_dn = np.concatenate(all_drug_names)
    global_r = pearson_r(pooled_t, pooled_p)
    mean_pdr = float(np.mean(list(per_drug_results.values()))) if per_drug_results else float("nan")

    logger.info(
        "%s: global_r=%.4f per_drug_r=%.4f n_drugs=%d",
        condition, global_r, mean_pdr, len(per_drug_results),
    )
    return {
        "global_r": global_r,
        "per_drug_r": mean_pdr,
        "per_drug_detail": per_drug_results,
        "n_focus_drugs": len(per_drug_results),
        "pooled_preds": pooled_p,
        "pooled_targets": pooled_t,
        "pooled_drug_names": pooled_dn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LINCS x within-MoA 2x2 factorial")
    parser.add_argument("--smoke", action="store_true", help="Quick run: 2 folds, ERK only")
    args = parser.parse_args()

    n_folds = 2 if args.smoke else K_FOLDS

    log_dir = EXP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logger.info("02_lincs_x_moa | ROOT=%s | folds=%d | smoke=%s", ROOT, n_folds, args.smoke)

    # --- Load omics ---
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    available_cells = set(rna.index) & set(mutations.index)

    # --- Build PASO drug set ---
    all_paso_drugs: set[str] = set()
    fold_data: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for k in range(K_FOLDS):
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, k
        )
        fold_data.append((train_df, test_df))
        all_paso_drugs |= set(train_df["drug_name"].unique()) | set(test_df["drug_name"].unique())
    logger.info("PASO drug set: %d drugs", len(all_paso_drugs))

    # Drug-to-test-fold mapping
    drug_to_test_fold: dict[str, int] = {}
    for k, (_, test_df) in enumerate(fold_data):
        for d in test_df["drug_name"].unique():
            drug_to_test_fold[d] = k

    # --- LINCS overlap ---
    with (DATA_DIR / "lincs_drug_index.json").open() as f:
        lincs_index = json.load(f)
    all_lincs_drugs = set(lincs_index["matched_drugs"])
    overlap_drugs = sorted(all_paso_drugs & all_lincs_drugs)
    overlap_set = set(overlap_drugs)
    logger.info(
        "Drug overlap: %d PASO x %d LINCS = %d overlap",
        len(all_paso_drugs), len(all_lincs_drugs), len(overlap_drugs),
    )

    # --- MoA annotations ---
    moa = load_moa_annotations()

    # Drug index over all PASO drugs (for LINCS feature matrix)
    drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(sorted(all_paso_drugs))}

    # --- LINCS PCA (fit on all overlap drugs, matching 04_external_signatures/01_lincs) ---
    lincs_raw, lincs_names = load_lincs_signatures(DATA_DIR, overlap_set)
    lincs_pca, var_explained = fit_lincs_pca(lincs_raw, LINCS_PCA_DIM)
    logger.info(
        "LINCS PCA(%d): %d drugs, variance explained = %.4f",
        LINCS_PCA_DIM, len(lincs_names), var_explained,
    )

    n_drugs = len(drug_to_idx)
    lincs_feat = np.zeros((n_drugs, lincs_pca.shape[1]), dtype=np.float32)
    for i, drug in enumerate(lincs_names):
        if drug in drug_to_idx:
            lincs_feat[drug_to_idx[drug]] = lincs_pca[i]

    # --- Identify focus drugs: in LINCS AND in a focus MoA ---
    focus_moas_used = FOCUS_MOAS[:1] if args.smoke else FOCUS_MOAS
    moa_groups_all = group_drugs_by_moa(list(all_paso_drugs), moa)

    # For each focus MoA, find LINCS-covered drugs
    moa_lincs_groups: dict[str, list[str]] = {}
    moa_stats: list[dict] = []
    for moa_label in focus_moas_used:
        moa_drugs = moa_groups_all.get(moa_label, [])
        lincs_covered = sorted([d for d in moa_drugs if d in overlap_set])
        moa_stats.append({
            "moa": moa_label,
            "n_drugs_total": len(moa_drugs),
            "n_drugs_lincs": len(lincs_covered),
        })
        if len(lincs_covered) >= MIN_MOA_DRUGS:
            moa_lincs_groups[moa_label] = lincs_covered
            logger.info(
                "MoA '%s': %d total, %d LINCS-covered — included",
                moa_label, len(moa_drugs), len(lincs_covered),
            )
        else:
            logger.warning(
                "MoA '%s': %d total, %d LINCS-covered — EXCLUDED (<3)",
                moa_label, len(moa_drugs), len(lincs_covered),
            )

    # Flat set of all focus drugs
    focus_drugs: set[str] = set()
    for drugs in moa_lincs_groups.values():
        focus_drugs.update(drugs)
    logger.info("Focus drugs (LINCS + focus MoA): %d drugs", len(focus_drugs))

    if not focus_drugs:
        logger.error("No focus drugs found — aborting")
        sys.exit(1)

    # ================================================================
    # Run 4 conditions
    # ================================================================
    results_4: dict[str, dict] = {}

    # --- 1. all_drug_no_lincs ---
    logger.info("=" * 60)
    logger.info("=== Condition: all_drug_no_lincs ===")
    results_4["all_drug_no_lincs"] = run_all_drug_folds(
        n_folds=n_folds, rna=rna, mutations=mutations,
        name_to_depmap=name_to_depmap, available_cells=available_cells,
        overlap_drugs=overlap_set, focus_drugs=focus_drugs,
        drug_to_idx=drug_to_idx, lincs_feat=None,
        condition="all_drug_no_lincs",
    )

    # --- 2. all_drug_lincs ---
    logger.info("=" * 60)
    logger.info("=== Condition: all_drug_lincs ===")
    results_4["all_drug_lincs"] = run_all_drug_folds(
        n_folds=n_folds, rna=rna, mutations=mutations,
        name_to_depmap=name_to_depmap, available_cells=available_cells,
        overlap_drugs=overlap_set, focus_drugs=focus_drugs,
        drug_to_idx=drug_to_idx, lincs_feat=lincs_feat,
        condition="all_drug_lincs",
    )

    # --- 3. within_moa_no_lincs ---
    logger.info("=" * 60)
    logger.info("=== Condition: within_moa_no_lincs ===")
    results_4["within_moa_no_lincs"] = run_within_moa(
        rna=rna, mutations=mutations,
        name_to_depmap=name_to_depmap, available_cells=available_cells,
        moa_lincs_groups=moa_lincs_groups,
        drug_to_test_fold=drug_to_test_fold,
        fold_data=fold_data,
        drug_to_idx=drug_to_idx, lincs_feat=None,
        overlap_set=overlap_set,
        condition="within_moa_no_lincs",
    )

    # --- 4. within_moa_lincs ---
    logger.info("=" * 60)
    logger.info("=== Condition: within_moa_lincs ===")
    results_4["within_moa_lincs"] = run_within_moa(
        rna=rna, mutations=mutations,
        name_to_depmap=name_to_depmap, available_cells=available_cells,
        moa_lincs_groups=moa_lincs_groups,
        drug_to_test_fold=drug_to_test_fold,
        fold_data=fold_data,
        drug_to_idx=drug_to_idx, lincs_feat=lincs_feat,
        overlap_set=overlap_set,
        condition="within_moa_lincs",
    )

    # ================================================================
    # Build output
    # ================================================================

    # Per-MoA factorial table
    per_moa_out: list[dict] = []
    for stat in moa_stats:
        moa_label = stat["moa"]
        moa_drugs_lincs = moa_lincs_groups.get(moa_label)
        if moa_drugs_lincs is None:
            per_moa_out.append({
                "moa": moa_label,
                "n_drugs_total": stat["n_drugs_total"],
                "n_drugs_lincs": stat["n_drugs_lincs"],
                "excluded": True,
                "reason": f"<{MIN_MOA_DRUGS} LINCS-covered drugs",
            })
            continue

        moa_drug_set = set(moa_drugs_lincs)
        factorial: dict[str, dict] = {}
        for cond_name, cond_res in results_4.items():
            # Per-drug r: filter to this MoA's drugs
            pdr_this_moa = {
                d: r for d, r in cond_res["per_drug_detail"].items()
                if d in moa_drug_set
            }
            moa_mean_pdr = float(np.mean(list(pdr_this_moa.values()))) if pdr_this_moa else float("nan")

            # Per-MoA global r: filter pooled arrays to this MoA's drugs
            pooled_p = cond_res["pooled_preds"]
            pooled_t = cond_res["pooled_targets"]
            pooled_dn = cond_res["pooled_drug_names"]
            if len(pooled_p) > 0:
                moa_mask = np.isin(pooled_dn, list(moa_drug_set))
                if moa_mask.sum() > 0:
                    moa_global_r = pearson_r(pooled_t[moa_mask], pooled_p[moa_mask])
                else:
                    moa_global_r = float("nan")
            else:
                moa_global_r = float("nan")

            factorial[cond_name] = {
                "global_r": round(moa_global_r, 6),
                "per_drug_r": round(moa_mean_pdr, 6),
            }

        per_moa_out.append({
            "moa": moa_label,
            "n_drugs_total": stat["n_drugs_total"],
            "n_drugs_lincs": stat["n_drugs_lincs"],
            "factorial": factorial,
        })

    # Per-drug detail table
    per_drug_out: list[dict] = []
    for drug in sorted(focus_drugs):
        drug_moa = moa.get(drug, "Unknown")
        drug_id_val = None  # PASO does not expose numeric drug IDs here
        row: dict = {
            "drug": drug,
            "drug_id": drug_id_val,
            "moa": drug_moa,
            "has_lincs": True,
        }
        for cond_name in ["all_drug_no_lincs", "within_moa_no_lincs",
                          "all_drug_lincs", "within_moa_lincs"]:
            r_val = results_4[cond_name]["per_drug_detail"].get(drug, float("nan"))
            row[f"{cond_name}_r"] = round(r_val, 6) if not np.isnan(r_val) else None
        per_drug_out.append(row)

    output = {
        "per_moa": per_moa_out,
        "per_drug": per_drug_out,
        "lincs_pca": {
            "n_components": LINCS_PCA_DIM,
            "variance_explained": round(var_explained, 4),
        },
        "n_folds": n_folds,
        "smoke": args.smoke,
    }

    # --- Write outputs ---
    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)

    results_path = report_data / "results.json"
    with results_path.open("w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", results_path)

    # Flat CSV: factorial_results.csv
    csv_rows: list[dict] = []
    for moa_entry in per_moa_out:
        if moa_entry.get("excluded"):
            continue
        for cond_name, metrics in moa_entry["factorial"].items():
            csv_rows.append({
                "moa": moa_entry["moa"],
                "condition": cond_name,
                "global_r": metrics["global_r"],
                "per_drug_r": metrics["per_drug_r"],
                "n_drugs": moa_entry["n_drugs_lincs"],
            })
    if csv_rows:
        csv_path = report_data / "factorial_results.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        logger.info("CSV written to %s", csv_path)

    # --- Summary ---
    logger.info("=" * 70)
    logger.info("2x2 FACTORIAL SUMMARY")
    logger.info("=" * 70)
    for moa_entry in per_moa_out:
        if moa_entry.get("excluded"):
            logger.info("MoA '%s': EXCLUDED (%s)", moa_entry["moa"], moa_entry.get("reason"))
            continue
        logger.info("MoA: %s (%d LINCS-covered drugs)", moa_entry["moa"], moa_entry["n_drugs_lincs"])
        logger.info("  %-25s  %8s  %10s", "Condition", "global_r", "per_drug_r")
        logger.info("  " + "-" * 50)
        for cond_name in ["all_drug_no_lincs", "all_drug_lincs",
                          "within_moa_no_lincs", "within_moa_lincs"]:
            m = moa_entry["factorial"][cond_name]
            logger.info("  %-25s  %8.4f  %10.4f", cond_name, m["global_r"], m["per_drug_r"])
        logger.info("")


if __name__ == "__main__":
    main()
