"""Oracle bounds for drug-feature null experiment.

Computes bounds that contextualize the drug-feature ablation.

Metric note
-----------
Experiments 02–07 use **mean per-drug Pearson r** (per-drug r averaged over test
drugs) as the primary metric. Bound 1 below uses **global Pearson r** instead,
because the drug-mean oracle assigns every cell the same prediction for a given
drug (its mean IC50), making per-drug r undefined (zero prediction variance).
Global r captures between-drug variation — a different but complementary axis.

The per-drug r speed-of-light is measured by Bounds 2 and 3: Pearson r between
matched drug profiles (same pathway or high Tanimoto) estimates how predictable
the within-drug cell-sensitivity ranking actually is.

  Bound 1 — Drug-mean oracle (global Pearson r):
    Predict every (cell, drug) pair as the drug's true mean IC50 (computed from
    the test pairs themselves). Reports global r — the ceiling for between-drug
    potency separation. Uses global r because predictions are constant per drug
    (per-drug r would be undefined; see metric note above).

  Bound 2 — Within-class profile concordance (per-drug r):
    For drugs sharing the same GDSC2 pathway annotation, compute Pearson r
    between their cell-line IC50 profiles on shared cell lines (≥5 required).
    Reports mean ± std across all within-class drug pairs.

  Bound 3 — Tanimoto-based concordance (per-drug r):
    For each drug, find nearest neighbor by Tanimoto similarity (Morgan FP,
    radius=2). Compute per-drug r between profiles. Reports mean for pairs
    with Tanimoto ≥ 0.7.

Output: EXP_DIR/report/data/metrics.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_drug_annotations(
    data_dir: Path,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Load GDSC2 drug annotations. Returns (df, col_name) or None if absent."""
    ann_path = data_dir / "gdsc2_drug_annotations.csv"
    if not ann_path.exists():
        logger.warning("Drug annotations not found: %s — skipping concordance", ann_path)
        return None

    df = pd.read_csv(ann_path)
    # Try to find the pathway/MoA column
    for col in ("pathway_name", "moa", "pathway", "MoA", "Pathway"):
        if col in df.columns:
            logger.info("Loaded drug annotations: %d rows, pathway col = %r", len(df), col)
            return df, col

    logger.warning(
        "Drug annotation file found but missing pathway/MoA column "
        "(tried: pathway_name, moa). Columns: %s — skipping concordance",
        list(df.columns),
    )
    return None


def build_response_pivot(dr_df: pd.DataFrame) -> pd.DataFrame:
    """Build cell × drug pivot from drug_response.parquet rows."""
    pivot = dr_df.pivot_table(
        index="depmap_id", columns="drug_name", values="ln_ic50", aggfunc="mean"
    )
    return pivot


def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity between two binary fingerprint vectors."""
    ab = float(np.dot(a, b))
    return ab / (a.sum() + b.sum() - ab + 1e-12)


def tanimoto_matrix(fps: np.ndarray) -> np.ndarray:
    """Compute n×n Tanimoto similarity matrix for binary fingerprint matrix (n, d)."""
    # T(a,b) = |a ∩ b| / |a ∪ b| = dot(a,b) / (|a| + |b| - dot(a,b))
    dot = fps @ fps.T  # (n, n)
    norms = fps.sum(axis=1)  # (n,)
    denom = norms[:, None] + norms[None, :] - dot  # (n, n)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return dot / denom


# ---------------------------------------------------------------------------
# Bound 1: Drug-mean oracle
# ---------------------------------------------------------------------------

def compute_drug_mean_oracle(
    paso_folds_dir: Path,
    k_folds: int = 5,
) -> Tuple[float, float, List[float]]:
    """For each test fold, predict (cell, drug) as drug's true mean IC50.

    Returns: (mean_r, std_r, fold_r_list)
    """
    fold_rs = []
    for fold_i in range(k_folds):
        test_df = pd.read_csv(paso_folds_dir / f"DrugBlind_test_Fold{fold_i}.csv")
        drug_means = test_df.groupby("drug")["IC50"].mean()
        preds = test_df["drug"].map(drug_means).values.astype(np.float64)
        targets = test_df["IC50"].values.astype(np.float64)

        mask = np.isfinite(preds) & np.isfinite(targets)
        r, _ = pearsonr(preds[mask], targets[mask])
        fold_rs.append(float(r))
        logger.info("Drug-mean oracle  fold %d: global r = %.4f  (n=%d)", fold_i, r, mask.sum())

    mean_r = float(np.mean(fold_rs))
    std_r = float(np.std(fold_rs))
    logger.info("Drug-mean oracle: mean=%.4f  std=%.4f", mean_r, std_r)
    return mean_r, std_r, fold_rs


# ---------------------------------------------------------------------------
# Bound 2: Within-class profile concordance
# ---------------------------------------------------------------------------

def compute_profile_concordance(
    pivot: pd.DataFrame,
    ann_df: pd.DataFrame,
    pathway_col: str,
    min_shared_cells: int = 5,
) -> Dict:
    """Compute per-drug r between same-pathway drug pairs on shared cells.

    Returns dict with mean, std, n_pairs, and per-pathway breakdown.
    """
    # Find the drug name column using explicit fallback list
    drug_col: Optional[str] = None
    for _c in ("drug_name", "Drug Name", "drug", "DRUG_NAME", "Drug"):
        if _c in ann_df.columns:
            drug_col = _c
            break
    if drug_col is None:
        logger.warning(
            "Drug annotation file has no recognizable drug name column "
            "(tried: drug_name, 'Drug Name', drug, DRUG_NAME, Drug). "
            "Columns: %s — skipping profile concordance",
            list(ann_df.columns),
        )
        return {
            "profile_concordance_mean": float("nan"),
            "profile_concordance_std": float("nan"),
            "profile_concordance_n_pairs": 0,
            "profile_concordance_by_pathway": {},
        }

    # Map drug → pathway
    drug_pathway: Dict[str, str] = {}
    for _, row in ann_df.iterrows():
        drug = row[drug_col]
        pathway = row[pathway_col]
        if pd.notna(drug) and pd.notna(pathway):
            drug_pathway[str(drug)] = str(pathway)

    # Build pathway → list of drugs (that are in our pivot)
    pivot_drugs = set(pivot.columns)
    pathway_drugs: Dict[str, List[str]] = {}
    for drug, pathway in drug_pathway.items():
        if drug in pivot_drugs:
            pathway_drugs.setdefault(pathway, []).append(drug)

    all_rs: List[float] = []
    by_pathway: Dict[str, Dict] = {}

    for pathway, drugs in pathway_drugs.items():
        if len(drugs) < 2:
            continue
        pathway_rs: List[float] = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                da, db = drugs[i], drugs[j]
                # Shared cells with non-nan values in both columns
                shared_mask = pivot[da].notna() & pivot[db].notna()
                if shared_mask.sum() < min_shared_cells:
                    continue
                va = pivot[da][shared_mask].values.astype(np.float64)
                vb = pivot[db][shared_mask].values.astype(np.float64)
                if va.std() < 1e-8 or vb.std() < 1e-8:
                    continue
                r, _ = pearsonr(va, vb)
                pathway_rs.append(float(r))
                all_rs.append(float(r))

        if pathway_rs:
            by_pathway[pathway] = {
                "mean": float(np.mean(pathway_rs)),
                "n_pairs": len(pathway_rs),
            }
            logger.info(
                "  pathway %-40s  %3d pairs  mean_r=%.3f",
                pathway[:40], len(pathway_rs), np.mean(pathway_rs),
            )

    if not all_rs:
        logger.warning("No within-class drug pairs found for concordance computation")
        return {
            "profile_concordance_mean": float("nan"),
            "profile_concordance_std": float("nan"),
            "profile_concordance_n_pairs": 0,
            "profile_concordance_by_pathway": {},
        }

    logger.info(
        "Profile concordance: %d pairs  mean=%.4f  std=%.4f",
        len(all_rs), np.mean(all_rs), np.std(all_rs),
    )
    return {
        "profile_concordance_mean": float(np.mean(all_rs)),
        "profile_concordance_std": float(np.std(all_rs)),
        "profile_concordance_n_pairs": len(all_rs),
        "profile_concordance_by_pathway": by_pathway,
    }


# ---------------------------------------------------------------------------
# Bound 3: Tanimoto-based concordance
# ---------------------------------------------------------------------------

def compute_tanimoto_concordance(
    pivot: pd.DataFrame,
    drug_to_idx: Dict[str, int],
    fp_matrix: np.ndarray,
    min_tanimoto: float = 0.7,
    min_shared_cells: int = 5,
) -> Dict:
    """For each drug, find nearest neighbor by Tanimoto; compute profile r."""
    pivot_drugs = [d for d in pivot.columns if d in drug_to_idx]
    if len(pivot_drugs) < 2:
        logger.warning("Too few drugs in pivot for Tanimoto concordance")
        return {
            "tanimoto_concordance_mean": float("nan"),
            "tanimoto_concordance_std": float("nan"),
            "tanimoto_concordance_n_pairs": 0,
        }

    idxs = [drug_to_idx[d] for d in pivot_drugs]
    fps = fp_matrix[idxs]  # (n_pivot_drugs, 2048)

    logger.info("Computing %dx%d Tanimoto matrix for pivot drugs ...", len(pivot_drugs), len(pivot_drugs))
    tani = tanimoto_matrix(fps.astype(np.float64))
    np.fill_diagonal(tani, 0.0)  # exclude self

    rs: List[float] = []
    for i, da in enumerate(pivot_drugs):
        # Nearest neighbor (excluding self)
        nn_j = int(np.argmax(tani[i]))
        sim = float(tani[i, nn_j])
        if sim < min_tanimoto:
            continue
        db = pivot_drugs[nn_j]
        shared_mask = pivot[da].notna() & pivot[db].notna()
        if shared_mask.sum() < min_shared_cells:
            continue
        va = pivot[da][shared_mask].values.astype(np.float64)
        vb = pivot[db][shared_mask].values.astype(np.float64)
        if va.std() < 1e-8 or vb.std() < 1e-8:
            continue
        r, _ = pearsonr(va, vb)
        rs.append(float(r))

    if not rs:
        logger.warning("No drug pairs with Tanimoto >= %.2f found", min_tanimoto)
        return {
            "tanimoto_concordance_mean": float("nan"),
            "tanimoto_concordance_std": float("nan"),
            "tanimoto_concordance_n_pairs": 0,
        }

    logger.info(
        "Tanimoto concordance (T>=%.2f): %d drugs  mean=%.4f  std=%.4f",
        min_tanimoto, len(rs), np.mean(rs), np.std(rs),
    )
    return {
        "tanimoto_concordance_mean": float(np.mean(rs)),
        "tanimoto_concordance_std": float(np.std(rs)),
        "tanimoto_concordance_n_pairs": len(rs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # File logger
    log_dir = EXP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logger.info("01_oracle_bounds | ROOT=%s", ROOT)
    logger.info("DATA_DIR=%s", DATA_DIR)

    # ---- Bound 1: Drug-mean oracle ----
    logger.info("=== Bound 1: Drug-mean oracle ===")
    oracle_mean, oracle_std, oracle_folds = compute_drug_mean_oracle(
        PASO_FOLDS_DIR, k_folds=K_FOLDS
    )

    # ---- Load drug response for concordance ----
    logger.info("=== Loading drug response matrix ===")
    dr_df = pd.read_parquet(DATA_DIR / "drug_response.parquet")
    pivot = build_response_pivot(dr_df)
    logger.info("Response pivot: %d cells × %d drugs", pivot.shape[0], pivot.shape[1])

    # ---- Load drug fingerprints for Tanimoto ----
    all_drugs = sorted(dr_df["drug_name"].unique())
    drug_to_idx: Dict[str, int] = {d: i for i, d in enumerate(all_drugs)}

    from src.data.drug_features import get_drug_fingerprints
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
    logger.info("Fingerprint matrix: %s", fp_matrix.shape)

    # ---- Bound 2: Within-class profile concordance ----
    concordance_result: Dict = {
        "profile_concordance_mean": float("nan"),
        "profile_concordance_std": float("nan"),
        "profile_concordance_n_pairs": 0,
        "profile_concordance_by_pathway": {},
    }
    ann_result = load_drug_annotations(DATA_DIR)
    if ann_result is not None:
        logger.info("=== Bound 2: Within-class profile concordance ===")
        ann_df, pathway_col = ann_result
        concordance_result = compute_profile_concordance(pivot, ann_df, pathway_col)
    else:
        logger.warning("Skipping Bound 2 (drug annotations absent)")

    # ---- Bound 3: Tanimoto concordance ----
    logger.info("=== Bound 3: Tanimoto-based concordance ===")
    tanimoto_result = compute_tanimoto_concordance(pivot, drug_to_idx, fp_matrix)

    # ---- Assemble and save ----
    metrics = {
        "oracle_global_r_mean": oracle_mean,
        "oracle_global_r_std": oracle_std,
        "oracle_global_r_folds": oracle_folds,
        **concordance_result,
        **tanimoto_result,
    }

    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)
    out_path = report_data / "metrics.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics written to %s", out_path)
    logger.info("oracle_global_r_mean=%.4f  oracle_global_r_std=%.4f", oracle_mean, oracle_std)
    if not np.isnan(concordance_result["profile_concordance_mean"]):
        logger.info(
            "profile_concordance_mean=%.4f  n_pairs=%d",
            concordance_result["profile_concordance_mean"],
            concordance_result["profile_concordance_n_pairs"],
        )
    if not np.isnan(tanimoto_result["tanimoto_concordance_mean"]):
        logger.info(
            "tanimoto_concordance_mean=%.4f  n_pairs=%d",
            tanimoto_result["tanimoto_concordance_mean"],
            tanimoto_result["tanimoto_concordance_n_pairs"],
        )


if __name__ == "__main__":
    main()
