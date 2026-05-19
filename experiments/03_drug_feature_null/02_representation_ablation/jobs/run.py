"""Ridge regression drug-feature representation ablation.

Tests whether any drug representation class improves within-drug cell-line
ranking in the drug-blind setting. All conditions use identical Ridge(α=1.0)
with RNA PCA(550) + mutation PCA(200) cell features and PASO 10-fold drug-blind CV.

Conditions:
  no_drug              — cell features only (baseline)
  morgan_fp_shuffled   — Morgan FP with drug-axis permuted (degenerate)
  random_continuous    — iid N(0,1) per drug, 2048-dim (degenerate)
  morgan_fp            — Morgan fingerprints 2048-bit
  chemberta            — ChemBERTa embeddings PCA(64); falls back to full 768-dim
  chembl_targets       — ChEMBL binary protein targets 5145-dim
  lincs                — L1000 signatures, matched drugs only
  prism                — PRISM pharmacological profiles, matched drugs only
  gnn                  — GNN embeddings 256-dim (requires gnn_embeddings_256.npy)
  all_concat           — morgan_fp + chemberta + chembl_targets concatenated

CLI:
  --skip CONDITION     skip one condition
  --only CONDITION     run only this condition (merges into existing metrics.json)

Output: EXP_DIR/report/data/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

from src.data.drug_features import get_drug_fingerprints
from src.evaluation.per_drug import per_drug_r
from src.evaluation.stats import bootstrap_delta_ci, holm_bonferroni
from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell, normalize_binary_fold, normalize_continuous_fold

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
RIDGE_ALPHA = 1.0
MIN_CELLS_PER_DRUG = 50
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 0
RNG_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_drug_annotations(data_dir: Path) -> Optional[Tuple[pd.DataFrame, str]]:
    """Load GDSC2 drug annotations. Returns (df, col_name) or None."""
    ann_path = data_dir / "gdsc2_drug_annotations.csv"
    if not ann_path.exists():
        return None
    df = pd.read_csv(ann_path)
    for col in ("moa", "pathway_name", "pathway", "MoA", "Pathway"):
        if col in df.columns:
            logger.info("Loaded drug annotations: %d rows, moa col = %r", len(df), col)
            return df, col
    logger.warning("Drug annotation file found but missing moa/pathway column")
    return None


def _all_gdsc_drugs(data_dir: Path) -> List[str]:
    """Sorted list of all GDSC drugs from drug_response.parquet."""
    dr = pd.read_parquet(data_dir / "drug_response.parquet")
    return sorted(dr["drug_name"].unique())


# ---------------------------------------------------------------------------
# Drug feature loaders
# ---------------------------------------------------------------------------

def _map_gdsc_to_paso(
    raw: np.ndarray,
    gdsc_drugs: List[str],
    drug_to_idx: Dict[str, int],
) -> np.ndarray:
    """Reindex a (n_gdsc_drugs, dim) array to (n_paso_drugs, dim) order."""
    gdsc_to_row = {d: i for i, d in enumerate(gdsc_drugs)}
    n = len(drug_to_idx)
    out = np.zeros((n, raw.shape[1]), dtype=np.float32)
    for drug, idx in drug_to_idx.items():
        row = gdsc_to_row.get(drug)
        if row is not None:
            out[idx] = raw[row].astype(np.float32)
    return out


def load_chemberta(data_dir: Path, drug_to_idx: Dict[str, int]) -> Optional[np.ndarray]:
    """Load ChemBERTa embeddings. Tries chemberta_pca64.npy, then drug_chembert_embeddings.npy."""
    gdsc_drugs = _all_gdsc_drugs(data_dir)
    for fname in ("chemberta_pca64.npy", "drug_chembert_embeddings.npy"):
        p = data_dir / fname
        if p.exists():
            raw = np.load(p)
            if raw.shape[0] != len(gdsc_drugs):
                logger.warning("%s shape[0]=%d != n_gdsc_drugs=%d — skipping", fname, raw.shape[0], len(gdsc_drugs))
                continue
            out = _map_gdsc_to_paso(raw, gdsc_drugs, drug_to_idx)
            logger.info("Loaded chemberta from %s: shape %s", fname, out.shape)
            return out
    logger.warning("chemberta data not found (tried chemberta_pca64.npy, drug_chembert_embeddings.npy)")
    return None


def load_chembl_targets(data_dir: Path, drug_to_idx: Dict[str, int]) -> Optional[np.ndarray]:
    """Load ChEMBL binary target features."""
    gdsc_drugs = _all_gdsc_drugs(data_dir)
    for fname in ("chembl_targets.npy", "drug_target_features.npy"):
        p = data_dir / fname
        if p.exists():
            raw = np.load(p)
            if raw.shape[0] != len(gdsc_drugs):
                logger.warning("%s shape[0]=%d != n_gdsc_drugs=%d — skipping", fname, raw.shape[0], len(gdsc_drugs))
                continue
            out = _map_gdsc_to_paso(raw, gdsc_drugs, drug_to_idx)
            logger.info("Loaded chembl_targets from %s: shape %s", fname, out.shape)
            return out
    logger.warning("chembl_targets not found (tried chembl_targets.npy, drug_target_features.npy)")
    return None


def load_lincs(
    data_dir: Path, drug_to_idx: Dict[str, int]
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Load LINCS signatures for matched drugs only.

    Returns (feat_matrix, matched_drug_names) where feat_matrix[i] corresponds
    to matched_drug_names[i], or None if unavailable.
    """
    idx_path = data_dir / "lincs_drug_index.json"

    # Try pca64 first, then full signatures
    for sig_fname in ("lincs_pca64.npy", "lincs_signatures.npy"):
        sig_path = data_dir / sig_fname
        if sig_path.exists() and idx_path.exists():
            raw = np.load(sig_path).astype(np.float32)
            with idx_path.open() as f:
                idx_data = json.load(f)
            matched = idx_data.get("matched_drugs", [])[:raw.shape[0]]
            keep = [(i, d) for i, d in enumerate(matched) if d in drug_to_idx]
            if not keep:
                logger.warning("%s: no matched drugs in PASO set", sig_fname)
                continue
            rows = [i for i, _ in keep]
            names = [d for _, d in keep]
            out = raw[rows]
            logger.info("Loaded %s: %d matched drugs, dim=%d", sig_fname, len(names), out.shape[1])
            return out, names

    logger.warning("LINCS data not found (tried lincs_pca64.npy, lincs_signatures.npy)")
    return None


def load_prism(
    data_dir: Path, drug_to_idx: Dict[str, int]
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Load PRISM profiles for matched drugs only."""
    for sig_fname, idx_fnames in [
        ("prism_pca64.npy", ["prism_drug_index.json", "prism_pca64_drugs.json"]),
    ]:
        sig_path = data_dir / sig_fname
        if not sig_path.exists():
            continue
        for idx_fname in idx_fnames:
            idx_path = data_dir / idx_fname
            if idx_path.exists():
                raw = np.load(sig_path).astype(np.float32)
                with idx_path.open() as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    drug_names = data.get("drugs", list(data.keys()))
                else:
                    drug_names = data
                drug_names = drug_names[:raw.shape[0]]
                keep = [(i, d) for i, d in enumerate(drug_names) if d in drug_to_idx]
                if not keep:
                    continue
                rows = [i for i, _ in keep]
                names = [d for _, d in keep]
                out = raw[rows]
                logger.info("Loaded %s: %d matched drugs, dim=%d", sig_fname, len(names), out.shape[1])
                return out, names

    logger.warning("PRISM data not found (tried prism_pca64.npy)")
    return None


def load_gnn(data_dir: Path, drug_to_idx: Dict[str, int]) -> Optional[np.ndarray]:
    """Load GNN embeddings (256-dim)."""
    gnn_path = data_dir / "gnn_embeddings_256.npy"
    if not gnn_path.exists():
        logger.warning("GNN embeddings not found: %s", gnn_path)
        return None
    raw = np.load(gnn_path).astype(np.float32)
    n = len(drug_to_idx)
    if raw.shape[0] == n:
        logger.info("Loaded gnn_embeddings_256 directly: shape %s", raw.shape)
        return raw
    gdsc_drugs = _all_gdsc_drugs(data_dir)
    if raw.shape[0] == len(gdsc_drugs):
        out = _map_gdsc_to_paso(raw, gdsc_drugs, drug_to_idx)
        logger.info("Loaded gnn_embeddings_256 (GDSC order): shape %s", out.shape)
        return out
    logger.warning(
        "gnn_embeddings_256.npy shape[0]=%d does not match n_paso=%d or n_gdsc=%d",
        raw.shape[0], n, len(gdsc_drugs),
    )
    return None


def tanimoto_matrix(fp: np.ndarray) -> np.ndarray:
    """Compute pairwise Tanimoto similarities for a binary fingerprint matrix.

    Tanimoto(A, B) = |A ∩ B| / |A ∪ B| = dot(A,B) / (|A| + |B| - dot(A,B)).
    Returns (n_drugs, n_drugs) float32 matrix.
    """
    fp = fp.astype(np.float32)
    dot = fp @ fp.T
    norms = fp.sum(axis=1, keepdims=True)  # (n, 1)
    union = norms + norms.T - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, dot / union, 0.0)
    return sim.astype(np.float32)


def max_train_tanimoto(
    test_drug_idxs: np.ndarray,
    train_drug_idxs: np.ndarray,
    sim_matrix: np.ndarray,
) -> np.ndarray:
    """For each test drug, return its maximum Tanimoto similarity to any training drug."""
    sub = sim_matrix[np.ix_(test_drug_idxs, train_drug_idxs)]
    return sub.max(axis=1)


def weighted_mean_r(rs: Dict[str, float], weights: Dict[str, int]) -> float:
    total_w = sum(weights.get(d, 1) for d in rs)
    if total_w == 0:
        return float("nan")
    return float(sum(rs[d] * weights.get(d, 1) for d in rs) / total_w)


# ---------------------------------------------------------------------------
# Single-fold Ridge run
# ---------------------------------------------------------------------------

def run_fold(
    fold_i: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    drug_feats_raw: Optional[np.ndarray],
    drug_to_idx: Dict[str, int],
    condition: str,
    ridge_alpha: float = RIDGE_ALPHA,
    min_cells: int = MIN_CELLS_PER_DRUG,
) -> Optional[Dict]:
    """Run one fold for one condition. Returns metrics dict or None on error."""
    # --- Cell features ---
    all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
    cell_to_row = {c: i for i, c in enumerate(all_cells)}
    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)
    train_cell_rows = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
    train_cell_set = np.unique(train_cell_rows)
    rna_c, mut_c = compress_cell(rna_arr, mut_arr, train_cell_set)
    cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

    # --- Drug feature normalization (per fold, fit on train drugs) ---
    train_drugs = sorted(train_df["drug_name"].unique())
    train_drug_idxs = np.array([drug_to_idx[d] for d in train_drugs], dtype=np.int32)
    drug_feat_norm: Optional[np.ndarray] = None
    if drug_feats_raw is not None:
        if condition in ("morgan_fp", "chembl_targets", "morgan_fp_shuffled"):
            drug_feat_norm, n_kept = normalize_binary_fold(drug_feats_raw, train_drug_idxs)
            logger.info(
                "  fold %d | %s: binary %d -> %d features after zero-var filter",
                fold_i, condition, drug_feats_raw.shape[1], n_kept,
            )
        elif condition == "all_concat":
            # drug_feats_raw already pre-normalized per block by caller
            drug_feat_norm = drug_feats_raw
        else:
            # Continuous: z-score
            drug_feat_norm = normalize_continuous_fold(drug_feats_raw, train_drug_idxs)

    # --- Filter test drugs: must have >= min_cells in test set ---
    test_cell_counts: Dict[str, int] = (
        test_df.groupby("drug_name")["depmap_id"].nunique().to_dict()
    )
    eligible = {d for d, n in test_cell_counts.items() if n >= min_cells}
    test_df_filt = pd.DataFrame(test_df[test_df["drug_name"].isin(eligible)])
    if test_df_filt.empty:
        logger.warning("fold %d | %s: no test drugs with >= %d cells", fold_i, condition, min_cells)
        return None

    # --- Build pair matrices ---
    def make_X(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows_c = np.array([cell_to_row[c] for c in df["depmap_id"]], dtype=np.int32)
        rows_d = np.array([drug_to_idx[d] for d in df["drug_name"]], dtype=np.int32)
        y = df["ln_ic50"].values.astype(np.float32)
        Xc = cell_feat[rows_c]
        if drug_feat_norm is not None:
            Xd = drug_feat_norm[rows_d]
            X = np.concatenate([Xc, Xd], axis=1)
        else:
            X = Xc
        return X, y, rows_d

    X_train, y_train, _ = make_X(train_df)
    X_test, y_test, _ = make_X(test_df_filt)
    drug_names_test = test_df_filt["drug_name"].values

    logger.info(
        "  fold %d | %s: train=%d  test=%d  eligible_drugs=%d  n_features=%d",
        fold_i, condition, len(y_train), len(y_test), len(eligible), X_train.shape[1],
    )

    # --- Fit Ridge ---
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
    preds = model.predict(X_test.astype(np.float64)).astype(np.float32)

    # --- Per-drug r ---
    per_drug = per_drug_r(preds, y_test, drug_names_test, min_cells=5)
    weights = {d: test_cell_counts[d] for d in per_drug}
    mean_r = float(np.mean(list(per_drug.values()))) if per_drug else float("nan")
    w_mean = weighted_mean_r(per_drug, weights)

    logger.info(
        "  fold %d | %s: mean_per_drug_r=%.4f  weighted=%.4f  n_drugs=%d",
        fold_i, condition, mean_r, w_mean, len(per_drug),
    )
    train_drug_set = sorted(train_df["drug_name"].unique())
    return {
        "mean_r": mean_r,
        "weighted_mean_r": w_mean,
        "per_drug_r": per_drug,
        "n_drugs": len(per_drug),
        "fold": fold_i,
        "train_drug_names": train_drug_set,
    }


# ---------------------------------------------------------------------------
# Build all_concat drug features for one fold (per-block normalization)
# ---------------------------------------------------------------------------

def build_all_concat_fold(
    fold_i: int,
    drug_to_idx: Dict[str, int],
    train_drugs: List[str],
    data_dir: Path,
) -> Optional[np.ndarray]:
    """Build all_concat with per-block normalization (fit on train drugs)."""
    train_drug_idxs = np.array([drug_to_idx[d] for d in train_drugs], dtype=np.int32)

    fp = get_drug_fingerprints(drug_to_idx, data_dir)
    chemberta = load_chemberta(data_dir, drug_to_idx)
    chembl = load_chembl_targets(data_dir, drug_to_idx)

    if chemberta is None or chembl is None:
        logger.warning("all_concat fold %d: missing chemberta or chembl_targets — skipping", fold_i)
        return None

    blocks: List[np.ndarray] = []

    # morgan_fp: binary
    fp_norm, n_fp = normalize_binary_fold(fp, train_drug_idxs)
    blocks.append(fp_norm)

    # chemberta: continuous z-score
    cb_norm = normalize_continuous_fold(chemberta, train_drug_idxs)
    blocks.append(cb_norm)

    # chembl_targets: binary
    ch_norm, n_ch = normalize_binary_fold(chembl, train_drug_idxs)
    blocks.append(ch_norm)

    feat = np.concatenate(blocks, axis=1)
    logger.info(
        "all_concat fold %d: fp(%d)+cb(%d)+chembl(%d) = %d total features",
        fold_i, n_fp, cb_norm.shape[1], n_ch, feat.shape[1],
    )
    return feat


# ---------------------------------------------------------------------------
# Run all folds for one condition
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    drug_to_idx: Dict[str, int],
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    name_to_depmap: Dict[str, str],
    data_dir: Path,
    paso_folds_dir: Path,
    k_folds: int = K_FOLDS,
    ridge_alpha: float = RIDGE_ALPHA,
) -> Optional[Dict]:
    """Run all folds for one condition. Returns aggregate metrics dict or None."""
    logger.info("=== Condition: %s (alpha=%.3g) ===", condition, ridge_alpha)

    # Load drug features once (except all_concat which is per-fold)
    if condition == "all_concat":
        drug_feats_preloaded: Optional[np.ndarray] = None  # built per fold
    else:
        drug_feats_preloaded = _load_condition_features(condition, drug_to_idx, data_dir)
        if drug_feats_preloaded is None and condition not in ("no_drug",):
            return None  # logged by loader

    # Restrict test set for lincs/prism to matched drugs only
    lincs_matched: Optional[set] = None
    prism_matched: Optional[set] = None
    if condition == "lincs":
        res = load_lincs(data_dir, drug_to_idx)
        if res is None:
            return None
        lincs_matched = set(res[1])
    if condition == "prism":
        res = load_prism(data_dir, drug_to_idx)
        if res is None:
            return None
        prism_matched = set(res[1])

    available_cells = set(rna.index) & set(mutations.index)
    fold_results: List[Dict] = []
    for fold_i in range(k_folds):
        train_df, test_df = load_paso_pairs(
            paso_folds_dir, name_to_depmap, available_cells, fold_i
        )
        train_df = pd.DataFrame(train_df[train_df["drug_name"].isin(drug_to_idx)])
        test_df = pd.DataFrame(test_df[test_df["drug_name"].isin(drug_to_idx)])
        if lincs_matched:
            test_df = pd.DataFrame(test_df[test_df["drug_name"].isin(lincs_matched)])
        if prism_matched:
            test_df = pd.DataFrame(test_df[test_df["drug_name"].isin(prism_matched)])
        if train_df.empty or test_df.empty:
            logger.warning("fold %d | %s: empty train or test", fold_i, condition)
            continue

        if condition == "all_concat":
            train_drugs = sorted(train_df["drug_name"].unique())
            drug_feats = build_all_concat_fold(fold_i, drug_to_idx, train_drugs, data_dir)
            if drug_feats is None:
                continue
        else:
            drug_feats = drug_feats_preloaded

        res = run_fold(
            fold_i=fold_i,
            train_df=train_df,
            test_df=test_df,
            rna=rna,
            mutations=mutations,
            drug_feats_raw=drug_feats,
            drug_to_idx=drug_to_idx,
            condition=condition,
            ridge_alpha=ridge_alpha,
        )
        if res is not None:
            fold_results.append(res)

    if not fold_results:
        logger.warning("Condition %s: all folds failed", condition)
        return None

    mean_rs = [r["mean_r"] for r in fold_results]
    w_mean_rs = [r["weighted_mean_r"] for r in fold_results]

    # Pool per-drug r across folds (drug-blind: each drug appears in one test fold)
    pooled_per_drug: Dict[str, float] = {}
    for r in fold_results:
        pooled_per_drug.update(r["per_drug_r"])

    return {
        "mean": float(np.mean(mean_rs)),
        "std": float(np.std(mean_rs)),
        "folds": [float(x) for x in mean_rs],
        "weighted_mean": float(np.mean(w_mean_rs)),
        "n_drugs_total": len(pooled_per_drug),
        "pooled_per_drug_r": pooled_per_drug,
        "fold_results": fold_results,
    }


def _load_condition_features(
    condition: str,
    drug_to_idx: Dict[str, int],
    data_dir: Path,
) -> Optional[np.ndarray]:
    """Load raw (unnormalized) drug features for a condition."""
    n = len(drug_to_idx)
    rng = np.random.default_rng(RNG_SEED)

    if condition == "no_drug":
        return None

    fp = get_drug_fingerprints(drug_to_idx, data_dir)

    if condition == "morgan_fp":
        return fp
    if condition == "morgan_fp_shuffled":
        perm = rng.permutation(n)
        return fp[perm]
    if condition == "random_continuous":
        return rng.standard_normal((n, 2048)).astype(np.float32)
    if condition == "chemberta":
        return load_chemberta(data_dir, drug_to_idx)
    if condition == "chembl_targets":
        return load_chembl_targets(data_dir, drug_to_idx)
    if condition == "gnn":
        return load_gnn(data_dir, drug_to_idx)
    if condition == "lincs":
        res = load_lincs(data_dir, drug_to_idx)
        if res is None:
            return None
        feats, matched = res
        out = np.zeros((n, feats.shape[1]), dtype=np.float32)
        for i, drug in enumerate(matched):
            if drug in drug_to_idx:
                out[drug_to_idx[drug]] = feats[i]
        return out
    if condition == "prism":
        res = load_prism(data_dir, drug_to_idx)
        if res is None:
            return None
        feats, matched = res
        out = np.zeros((n, feats.shape[1]), dtype=np.float32)
        for i, drug in enumerate(matched):
            if drug in drug_to_idx:
                out[drug_to_idx[drug]] = feats[i]
        return out
    if condition == "all_concat":
        return None  # handled per-fold

    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Drug feature representation ablation")
    parser.add_argument("--skip", metavar="CONDITION", default=None)
    parser.add_argument("--only", metavar="CONDITION", default=None)
    parser.add_argument("--no-alpha-sensitivity", action="store_true")
    args = parser.parse_args()

    log_dir = EXP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logger.info("02_representation_ablation | ROOT=%s", ROOT)

    # Load omics
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)

    # Build drug index from PASO folds 0-4
    all_drugs: set[str] = set()
    for fold_i in range(K_FOLDS):
        tr = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{fold_i}.csv")
        te = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{fold_i}.csv")
        all_drugs |= set(tr["drug"].unique()) | set(te["drug"].unique())
    drug_to_idx: Dict[str, int] = {d: i for i, d in enumerate(sorted(all_drugs))}
    logger.info("PASO drug set: %d drugs", len(drug_to_idx))

    # Conditions
    all_conditions = [
        "no_drug", "morgan_fp_shuffled", "random_continuous",
        "morgan_fp", "chemberta", "chembl_targets",
        "lincs", "prism", "gnn", "all_concat",
    ]
    non_degenerate = [
        "morgan_fp", "chemberta", "chembl_targets", "lincs",
        "prism", "gnn", "all_concat",
    ]

    if args.only:
        conditions_to_run = [args.only]
    elif args.skip:
        conditions_to_run = [c for c in all_conditions if c != args.skip]
    else:
        conditions_to_run = all_conditions

    logger.info("Conditions: %s", conditions_to_run)

    # Load existing output for merge mode
    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)
    metrics_path = report_data / "metrics.json"
    progress_path = report_data / "progress.json"

    existing: Dict = {}
    if args.only and metrics_path.exists():
        with metrics_path.open() as f:
            existing = json.load(f)
        logger.info("Loaded existing metrics from %s", metrics_path)

    # Storage: summary results and pooled per-drug r (needed for delta/CI)
    condition_summary: Dict[str, Dict] = dict(existing.get("conditions", {}))

    # Load sidecar so that --only CONDITION runs still see other conditions' pooled r
    pooled_sidecar_path = report_data / "pooled_per_drug_r.json"
    pooled_all: Dict[str, Dict[str, float]] = {}
    if pooled_sidecar_path.exists():
        with pooled_sidecar_path.open() as f:
            pooled_all = json.load(f)
        logger.info("Loaded pooled_per_drug_r sidecar: %d conditions", len(pooled_all))

    # Run each condition once, storing both summary and per-drug r
    for condition in conditions_to_run:
        res = run_condition(
            condition=condition,
            drug_to_idx=drug_to_idx,
            rna=rna,
            mutations=mutations,
            name_to_depmap=name_to_depmap,
            data_dir=DATA_DIR,
            paso_folds_dir=PASO_FOLDS_DIR,
        )
        if res is not None:
            pooled_all[condition] = res["pooled_per_drug_r"]
            condition_summary[condition] = {
                "mean": res["mean"],
                "std": res["std"],
                "folds": res["folds"],
                "weighted_mean": res["weighted_mean"],
                "n_drugs": res["n_drugs_total"],
            }
            # Persist pooled_per_drug_r sidecar so --only CONDITION runs can see all conditions
            with pooled_sidecar_path.open("w") as f:
                json.dump(pooled_all, f)
        # Checkpoint
        with progress_path.open("w") as f:
            json.dump({"conditions": condition_summary}, f, indent=2)

    # --- Compute Δ, p-values, Holm correction ---
    no_drug_pooled = pooled_all.get("no_drug", {})

    # Per-drug delta: morgan_fp − no_drug (matched drugs)
    per_drug_delta: Dict[str, float] = {}
    morgan_pooled = pooled_all.get("morgan_fp", {})
    for drug in sorted(set(no_drug_pooled) & set(morgan_pooled)):
        per_drug_delta[drug] = morgan_pooled[drug] - no_drug_pooled[drug]

    # T-test each non-degenerate condition vs no_drug
    p_values_raw: Dict[str, float] = {}
    for condition in non_degenerate:
        pooled = pooled_all.get(condition, {})
        common = sorted(set(no_drug_pooled) & set(pooled))
        if len(common) < 3:
            continue
        diffs = np.array([pooled[d] - no_drug_pooled[d] for d in common])
        _, p = ttest_1samp(diffs, 0.0)
        p_values_raw[condition] = float(p)  # type: ignore[arg-type]

    # Apply Holm-Bonferroni only to conditions that were actually run
    conditions_for_holm = [c for c in non_degenerate if c in p_values_raw]
    logger.info("Holm-Bonferroni over %d conditions: %s", len(conditions_for_holm), conditions_for_holm)
    holm_ps = holm_bonferroni({c: p_values_raw[c] for c in conditions_for_holm})

    # Bootstrap CI on morgan_fp Δ
    ci_lo, ci_hi = float("nan"), float("nan")
    if per_drug_delta:
        ci_lo, ci_hi = bootstrap_delta_ci(per_drug_delta)
        logger.info(
            "Bootstrap CI for morgan_fp Δ: [%.4f, %.4f]  mean=%.4f  n=%d",
            ci_lo, ci_hi,
            float(np.mean(list(per_drug_delta.values()))),
            len(per_drug_delta),
        )

    # Annotate condition_summary with delta / holm_p / ci_95
    for condition in list(condition_summary.keys()):
        if condition == "no_drug":
            continue
        pooled = pooled_all.get(condition, {})
        common = sorted(set(no_drug_pooled) & set(pooled))
        if common:
            condition_summary[condition]["delta"] = round(
                float(np.mean([pooled[d] for d in common])) -
                float(np.mean([no_drug_pooled[d] for d in common])),
                6,
            )
        if condition in holm_ps:
            condition_summary[condition]["holm_p"] = float(holm_ps[condition])
            condition_summary[condition]["raw_p"] = float(p_values_raw[condition])
        if condition == "morgan_fp" and not np.isnan(ci_lo):
            condition_summary[condition]["ci_95"] = [float(ci_lo), float(ci_hi)]

    # --- Alpha sensitivity for morgan_fp ---
    alpha_sensitivity: Dict = {}
    if not args.no_alpha_sensitivity and "morgan_fp" in conditions_to_run:
        logger.info("=== Alpha sensitivity check (global α) ===")
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            res = run_condition(
                "morgan_fp", drug_to_idx, rna, mutations, name_to_depmap,
                DATA_DIR, PASO_FOLDS_DIR, ridge_alpha=alpha,
            )
            if res is not None:
                alpha_sensitivity[str(alpha)] = {"mean": res["mean"], "std": res["std"]}
                logger.info("alpha=%.4g  mean_per_drug_r=%.4f", alpha, res["mean"])

    # --- Per-block drug scale sweep ---
    # Multiplying drug features by c changes the effective relative regularization
    # between cell (750-dim dense) and drug (2048-dim sparse binary) blocks.
    # c < 1 shrinks drug contributions relative to cell; c > 1 amplifies them.
    # If the null is a regularization artifact (cell block dominates), we expect
    # Δ > 0.01 to appear at some c > 1.
    drug_scale_sweep: Dict = {}
    if not args.no_alpha_sensitivity and "morgan_fp" in pooled_all:
        logger.info("=== Per-block drug scale sweep ===")
        fp_all = get_drug_fingerprints(drug_to_idx, DATA_DIR)
        for c in [0.1, 0.3, 1.0, 3.0, 10.0]:
            scaled_fp = (fp_all * c).astype(np.float32)
            # Run with no_drug baseline first, then morgan_fp scaled
            res_nd = run_condition(
                "no_drug", drug_to_idx, rna, mutations, name_to_depmap,
                DATA_DIR, PASO_FOLDS_DIR,
            )
            # Temporarily override fp features by running condition with pre-scaled features.
            # We pass scaled_fp as a pre-loaded feature array by constructing a one-off loop.
            fold_rs = []
            available_cells_sc = set(rna.index) & set(mutations.index)
            for fold_i in range(K_FOLDS):
                tr_df, te_df = load_paso_pairs(
                    PASO_FOLDS_DIR, name_to_depmap, available_cells_sc, fold_i
                )
                tr_df = pd.DataFrame(tr_df[tr_df["drug_name"].isin(drug_to_idx)])
                te_df = pd.DataFrame(te_df[te_df["drug_name"].isin(drug_to_idx)])
                if tr_df.empty or te_df.empty:
                    continue
                r = run_fold(
                    fold_i=fold_i,
                    train_df=tr_df,
                    test_df=te_df,
                    rna=rna,
                    mutations=mutations,
                    drug_feats_raw=scaled_fp,
                    drug_to_idx=drug_to_idx,
                    condition="morgan_fp",
                    ridge_alpha=RIDGE_ALPHA,
                )
                if r is not None:
                    fold_rs.append(r["mean_r"])
            if fold_rs and res_nd is not None:
                mean_fp = float(np.mean(fold_rs))
                delta_c = mean_fp - res_nd["mean"]
                drug_scale_sweep[str(c)] = {
                    "mean": mean_fp,
                    "std": float(np.std(fold_rs)),
                    "delta": float(delta_c),
                }
                logger.info(
                    "drug_scale=%.1f  morgan_fp_mean=%.4f  Δ=%.4f",
                    c, mean_fp, delta_c,
                )

    # --- Similarity-stratified Δ analysis ---
    # For each test drug, compute max Tanimoto similarity to any training drug in that fold.
    # Bin into low(<0.3), mid(0.3–0.5), high(≥0.5) and report Δ per bin.
    # Rationale: Morgan FP can only transfer signal to structurally similar drugs.
    # If the aggregate null holds even in the high-similarity bin, the null is robust.
    similarity_stratified: Dict = {}
    if "morgan_fp" in pooled_all and "no_drug" in pooled_all:
        fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
        sim_mat = tanimoto_matrix(fp_matrix)
        drug_max_sim: Dict[str, float] = {}
        available_cells_sim = set(rna.index) & set(mutations.index)
        for fold_i in range(K_FOLDS):
            tr_df, te_df = load_paso_pairs(
                PASO_FOLDS_DIR, name_to_depmap, available_cells_sim, fold_i
            )
            tr_df = pd.DataFrame(tr_df[tr_df["drug_name"].isin(drug_to_idx)])
            te_df = pd.DataFrame(te_df[te_df["drug_name"].isin(drug_to_idx)])
            if tr_df.empty or te_df.empty:
                continue
            train_drug_idxs = np.array(
                [drug_to_idx[d] for d in sorted(tr_df["drug_name"].unique())],
                dtype=np.int32,
            )
            test_drugs_fold = sorted(te_df["drug_name"].unique())
            test_drug_idxs = np.array([drug_to_idx[d] for d in test_drugs_fold], dtype=np.int32)
            max_sims = max_train_tanimoto(test_drug_idxs, train_drug_idxs, sim_mat)
            for drug, s in zip(test_drugs_fold, max_sims):
                drug_max_sim[drug] = float(s)

        bins = {"low": (0.0, 0.3), "mid": (0.3, 0.5), "high": (0.5, 1.01)}
        for bin_name, (lo, hi) in bins.items():
            drugs_in_bin = [
                d for d, s in drug_max_sim.items()
                if lo <= s < hi
                and d in no_drug_pooled
                and d in pooled_all.get("morgan_fp", {})
            ]
            if not drugs_in_bin:
                continue
            nd_vals = [no_drug_pooled[d] for d in drugs_in_bin]
            fp_vals = [pooled_all["morgan_fp"][d] for d in drugs_in_bin]
            delta_bin = float(np.mean(fp_vals)) - float(np.mean(nd_vals))
            similarity_stratified[bin_name] = {
                "n_drugs": len(drugs_in_bin),
                "no_drug_mean": round(float(np.mean(nd_vals)), 6),
                "morgan_fp_mean": round(float(np.mean(fp_vals)), 6),
                "delta": round(delta_bin, 6),
                "sim_range": [lo, hi],
            }
            logger.info(
                "Similarity bin %-4s (%.1f–%.1f): n=%d  no_drug=%.4f  morgan_fp=%.4f  Δ=%.4f",
                bin_name, lo, hi, len(drugs_in_bin),
                float(np.mean(nd_vals)), float(np.mean(fp_vals)), delta_bin,
            )
        if drug_max_sim:
            sims = list(drug_max_sim.values())
            logger.info(
                "Tanimoto NN to train: mean=%.3f  median=%.3f  pct_high=%.1f%%",
                float(np.mean(sims)), float(np.median(sims)),
                100.0 * sum(s >= 0.5 for s in sims) / len(sims),
            )

    # --- Per-MoA delta ---
    per_moa_delta: Dict[str, Dict] = {}
    ann_result = load_drug_annotations(DATA_DIR)
    if ann_result is not None and per_drug_delta:
        ann_df, moa_col = ann_result
        drug_moa: Dict[str, str] = {}
        # Find the drug name column using explicit fallback list
        drug_col = None
        for _c in ("drug_name", "Drug Name", "drug", "DRUG_NAME", "Drug"):
            if _c in ann_df.columns:
                drug_col = _c
                break
        if drug_col is None:
            logger.warning(
                "Drug annotation file has no recognizable drug name column "
                "(tried: drug_name, 'Drug Name', drug, DRUG_NAME, Drug). "
                "Columns: %s — skipping per-MoA delta",
                list(ann_df.columns),
            )
        if drug_col:
            for _, row in ann_df.iterrows():
                if pd.notna(row[drug_col]) and pd.notna(row[moa_col]):
                    drug_moa[str(row[drug_col])] = str(row[moa_col])
        moa_groups: Dict[str, List[float]] = {}
        for drug, delta in per_drug_delta.items():
            moa = drug_moa.get(drug, "Unknown")
            moa_groups.setdefault(moa, []).append(delta)
        per_moa_delta = {
            moa: {"delta_mean": float(np.mean(vals)), "n_drugs": len(vals)}
            for moa, vals in moa_groups.items()
        }

    # --- Write final output ---
    output = {
        "conditions": condition_summary,
        "per_drug_delta": per_drug_delta,
        "per_moa_delta": per_moa_delta,
        "alpha_sensitivity_morgan_fp": alpha_sensitivity,
        "drug_scale_sweep": drug_scale_sweep,
        "similarity_stratified_delta": similarity_stratified,
    }
    with metrics_path.open("w") as f:
        json.dump(output, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)

    # Summary table
    logger.info("=" * 70)
    logger.info("%-25s  %8s  %8s  %8s  %8s", "Condition", "mean_r", "Δ", "holm_p", "n_drugs")
    for cond, data in condition_summary.items():
        delta = data.get("delta", float("nan"))
        hp = data.get("holm_p", float("nan"))
        nd = data.get("n_drugs", 0)
        logger.info("%-25s  %8.4f  %8.4f  %8.4f  %8d", cond, data["mean"], delta, hp, nd)


if __name__ == "__main__":
    main()
