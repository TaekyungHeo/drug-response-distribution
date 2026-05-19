"""04_split_robustness: scaffold-blind 5-fold Ridge ablation (morgan_fp vs no_drug).

Replaces PASO random drug-blind folds with Bemis-Murcko scaffold-stratified folds.
Every drug sharing a scaffold is held out together, making test drugs structurally
novel relative to the training set. If the null (morgan_fp Δ ≈ 0) holds here,
it is not a split-design artifact.

Output: report/data/metrics.json
"""

from __future__ import annotations

import heapq
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))

from src.data.drug_features import get_drug_fingerprints
from src.evaluation.per_drug import mean_per_drug_r
from src.utils.ridge import compress_cell, normalize_binary_fold, safe_fit_scaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"
EXP_DIR = Path(__file__).parents[1]
K_FOLDS = 5
MIN_CELLS_PER_DRUG = 50
MIN_CELLS_EVAL = 5
RIDGE_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Scaffold assignment
# ---------------------------------------------------------------------------

def compute_scaffolds(smiles_dict: Dict[str, Optional[str]], drug_names: List[str]) -> Dict[str, str]:
    """Compute Bemis-Murcko scaffold string for each drug with valid SMILES.

    Returns:
        Dict mapping drug_name -> canonical scaffold SMILES string.
        Drugs with null/invalid SMILES are NOT included.
    """
    try:
        from rdkit import Chem  # type: ignore[import-untyped]
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("rdkit required for scaffold computation: pip install rdkit")

    scaffolds: Dict[str, str] = {}
    for drug in drug_names:
        smi = smiles_dict.get(drug)
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold_smi = MurckoScaffoldSmiles(mol=mol)
        if scaffold_smi == "":
            # No ring system: acyclic drug — use canonical SMILES as singleton scaffold
            canonical = Chem.MolToSmiles(mol)
            scaffolds[drug] = f"__acyclic__{canonical}"
        else:
            scaffolds[drug] = scaffold_smi
    return scaffolds


def scaffold_5fold_assignment(
    scaffold_to_drugs: Dict[str, List[str]],
    n_folds: int = 5,
) -> List[List[str]]:
    """Greedy bin-packing scaffold assignment to folds.

    Sorts scaffold groups by size descending, assigns each group to the fold
    with the fewest drugs (min-heap). Returns list of n_folds drug lists.
    """
    # Sort scaffold groups by size descending
    groups = sorted(scaffold_to_drugs.values(), key=len, reverse=True)

    # Min-heap: (current_fold_size, fold_index)
    heap: List[Tuple[int, int]] = [(0, i) for i in range(n_folds)]
    heapq.heapify(heap)

    fold_drugs: List[List[str]] = [[] for _ in range(n_folds)]
    for group in groups:
        size, fold_idx = heapq.heappop(heap)
        fold_drugs[fold_idx].extend(group)
        heapq.heappush(heap, (size + len(group), fold_idx))

    return fold_drugs


def verify_scaffold_no_leak(
    fold_drugs: List[List[str]],
    drug_to_scaffold: Dict[str, str],
) -> None:
    """Assert train scaffolds ∩ test scaffolds == ∅ for every fold (hard failure)."""
    for k in range(len(fold_drugs)):
        test_drugs = set(fold_drugs[k])
        train_drugs = set(d for i, drugs in enumerate(fold_drugs) if i != k for d in drugs)
        test_scaffolds = {drug_to_scaffold[d] for d in test_drugs if d in drug_to_scaffold}
        train_scaffolds = {drug_to_scaffold[d] for d in train_drugs if d in drug_to_scaffold}
        leak = test_scaffolds & train_scaffolds
        assert len(leak) == 0, (
            f"Scaffold leak in fold {k}! {len(leak)} scaffolds appear in both train and test: "
            f"{list(leak)[:3]}"
        )
    logger.info("Scaffold leak assertion PASSED for all %d folds", len(fold_drugs))


# ---------------------------------------------------------------------------
# Ridge fold runner
# ---------------------------------------------------------------------------

def run_fold(
    k: int,
    fold_drugs: List[List[str]],
    dr: pd.DataFrame,
    rna: pd.DataFrame,
    mutations: pd.DataFrame,
    fp_matrix: np.ndarray,
    drug_to_idx: Dict[str, int],
) -> Tuple[float, float]:
    """Fit Ridge for fold k, return (no_drug_r, morgan_fp_r)."""
    test_drug_set = set(fold_drugs[k])
    train_drug_set = set(d for i, drugs in enumerate(fold_drugs) if i != k for d in drugs)

    available_cells = set(rna.index) & set(mutations.index)

    train_df = dr[dr["drug_name"].isin(train_drug_set) & dr["depmap_id"].isin(available_cells)].copy()
    test_df_full = dr[dr["drug_name"].isin(test_drug_set) & dr["depmap_id"].isin(available_cells)].copy()

    # Exclude test drugs with < MIN_CELLS_PER_DRUG test pairs
    test_counts = test_df_full.groupby("drug_name").size()
    valid_test_drugs = test_counts[test_counts >= MIN_CELLS_PER_DRUG].index
    test_df = test_df_full[test_df_full["drug_name"].isin(valid_test_drugs)].copy()

    logger.info(
        "  Fold %d: train_drugs=%d test_drugs=%d (of %d; %d excluded <50 cells) "
        "train_pairs=%d test_pairs=%d",
        k,
        len(train_drug_set),
        len(valid_test_drugs),
        len(test_drug_set),
        len(test_drug_set) - len(valid_test_drugs),
        len(train_df),
        len(test_df),
    )

    all_cells_train = sorted(train_df["depmap_id"].unique())
    all_cells_test = sorted(test_df["depmap_id"].unique())
    all_cells = sorted(set(all_cells_train) | set(all_cells_test))

    # Build cell-row lookup
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mutations.loc[all_cells].values.astype(np.float32)

    # PCA fit on training cells only
    train_cell_rows = np.array([cell_to_row[c] for c in all_cells_train], dtype=np.int32)
    rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_cell_rows, rna_dim=550, mut_dim=200)
    cell_feat = np.concatenate([rna_pca, mut_pca], axis=1)  # (n_all_cells, 750)

    # Build train and test index arrays
    train_cell_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
    test_cell_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
    train_drug_idx = np.array([drug_to_idx[d] for d in train_df["drug_name"]], dtype=np.int32)
    test_drug_idx = np.array([drug_to_idx[d] for d in test_df["drug_name"]], dtype=np.int32)
    y_train = train_df["ln_ic50"].values.astype(np.float32)
    y_test = test_df["ln_ic50"].values.astype(np.float32)
    test_drug_names = test_df["drug_name"].values

    X_cell_train = cell_feat[train_cell_idx]
    X_cell_test = cell_feat[test_cell_idx]

    # --- no_drug condition ---
    sc_nd = safe_fit_scaler(X_cell_train)
    X_train_nd = sc_nd.transform(X_cell_train)
    X_test_nd = sc_nd.transform(X_cell_test)
    ridge_nd = Ridge(alpha=RIDGE_ALPHA)
    ridge_nd.fit(X_train_nd, y_train)
    preds_nd = ridge_nd.predict(X_test_nd)
    r_no_drug = mean_per_drug_r(preds_nd, y_test, test_drug_names, min_cells=MIN_CELLS_EVAL)

    # --- morgan_fp condition ---
    # Cell features: z-score; Drug features: drop zero-var bits only (binary, not z-scored)
    sc_cell = safe_fit_scaler(X_cell_train)
    X_cell_train_sc = sc_cell.transform(X_cell_train)
    X_cell_test_sc = sc_cell.transform(X_cell_test)
    train_drug_unique = np.unique(train_drug_idx)
    fp_norm, _ = normalize_binary_fold(fp_matrix, train_drug_unique)
    X_train_fp = np.concatenate([X_cell_train_sc, fp_norm[train_drug_idx]], axis=1)
    X_test_fp = np.concatenate([X_cell_test_sc, fp_norm[test_drug_idx]], axis=1)
    ridge_fp = Ridge(alpha=RIDGE_ALPHA)
    ridge_fp.fit(X_train_fp, y_train)
    preds_fp = ridge_fp.predict(X_test_fp)
    r_morgan_fp = mean_per_drug_r(preds_fp, y_test, test_drug_names, min_cells=MIN_CELLS_EVAL)

    logger.info("  Fold %d: no_drug=%.4f  morgan_fp=%.4f  delta=%.4f",
                k, r_no_drug, r_morgan_fp, r_morgan_fp - r_no_drug)
    return r_no_drug, r_morgan_fp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("04_split_robustness: scaffold-blind 5-fold Ridge ablation")

    # Load omics data
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  Mutations: %s", rna.shape, mutations.shape)

    # Build drug set from PASO folds — consistent with 02_representation_ablation.
    # This gives exactly 233 drugs with valid SMILES, matching the fingerprint cache.
    paso_drug_set: set[str] = set()
    for fold_i in range(10):
        tr = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_train_Fold{fold_i}.csv")
        te = pd.read_csv(PASO_FOLDS_DIR / f"DrugBlind_test_Fold{fold_i}.csv")
        paso_drug_set |= set(tr["drug"].unique()) | set(te["drug"].unique())
    drug_names_all = sorted(paso_drug_set)
    logger.info("Drugs (PASO set): %d", len(drug_names_all))

    # Build drug index and load all PASO pairs from drug_response.parquet
    available_cells = set(rna.index) & set(mutations.index)
    dr = pd.read_parquet(DATA_DIR / "drug_response.parquet")
    dr = pd.DataFrame(
        dr[dr["depmap_id"].isin(available_cells) & dr["drug_name"].isin(paso_drug_set)]
    ).reset_index(drop=True)
    logger.info("Drug response (PASO drugs × available cells): %d pairs, %d drugs, %d cells",
                len(dr), dr["drug_name"].nunique(), dr["depmap_id"].nunique())

    # Build drug index
    drug_to_idx = {d: i for i, d in enumerate(drug_names_all)}

    # Get Morgan fingerprints
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
    logger.info("Fingerprint matrix: %s", fp_matrix.shape)

    # Load SMILES
    smiles_path = DATA_DIR / "drug_smiles.json"
    if not smiles_path.exists():
        raise FileNotFoundError(f"SMILES cache not found: {smiles_path}")
    with smiles_path.open() as f:
        smiles_dict_full: Dict[str, Optional[str]] = json.load(f)

    # Filter SMILES dict to drugs actually in our response data
    smiles_dict = {d: smiles_dict_full.get(d) for d in drug_names_all}
    n_null = sum(1 for v in smiles_dict.values() if v is None)
    n_invalid = 0
    try:
        from rdkit import Chem  # type: ignore[import-untyped]
        for _, smi in smiles_dict.items():
            if smi is not None and Chem.MolFromSmiles(smi) is None:
                n_invalid += 1
    except ImportError:
        pass
    n_excluded = n_null + n_invalid
    logger.info("SMILES: %d null, %d invalid, %d total excluded (of %d drugs)",
                n_null, n_invalid, n_excluded, len(drug_names_all))
    if n_excluded > 5:
        raise RuntimeError(
            f"{n_excluded} drugs excluded due to missing/invalid SMILES (threshold: 5). "
            "Investigate data quality before proceeding."
        )

    # Compute scaffolds for valid drugs
    logger.info("Computing Bemis-Murcko scaffolds...")
    drug_to_scaffold = compute_scaffolds(smiles_dict, drug_names_all)
    logger.info("Scaffolds computed for %d / %d drugs", len(drug_to_scaffold), len(drug_names_all))

    # Group drugs by scaffold
    scaffold_to_drugs: Dict[str, List[str]] = {}
    for drug, scaffold in drug_to_scaffold.items():
        scaffold_to_drugs.setdefault(scaffold, []).append(drug)

    n_scaffolds = len(scaffold_to_drugs)
    logger.info("Unique scaffolds: %d", n_scaffolds)

    # Assign scaffold groups to 5 folds by greedy bin-packing
    fold_drugs = scaffold_5fold_assignment(scaffold_to_drugs, n_folds=K_FOLDS)
    fold_sizes = [len(f) for f in fold_drugs]
    logger.info("Fold sizes: %s (total=%d)", fold_sizes, sum(fold_sizes))

    # Verify scaffold no-leak
    verify_scaffold_no_leak(fold_drugs, drug_to_scaffold)

    # Run 5-fold CV
    no_drug_folds: List[float] = []
    morgan_fp_folds: List[float] = []

    for k in range(K_FOLDS):
        logger.info("=== Fold %d/%d ===", k, K_FOLDS - 1)
        r_nd, r_fp = run_fold(k, fold_drugs, dr, rna, mutations, fp_matrix, drug_to_idx)
        no_drug_folds.append(r_nd)
        morgan_fp_folds.append(r_fp)

    # Summary
    nd_mean = float(np.mean(no_drug_folds))
    nd_std = float(np.std(no_drug_folds))
    fp_mean = float(np.mean(morgan_fp_folds))
    fp_std = float(np.std(morgan_fp_folds))
    delta = fp_mean - nd_mean

    logger.info("=" * 60)
    logger.info("no_drug:   mean=%.4f ± %.4f  folds=%s",
                nd_mean, nd_std, [round(r, 4) for r in no_drug_folds])
    logger.info("morgan_fp: mean=%.4f ± %.4f  folds=%s  delta=%.4f",
                fp_mean, fp_std, [round(r, 4) for r in morgan_fp_folds], delta)

    # Write output
    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "no_drug": {
            "mean": nd_mean,
            "std": nd_std,
            "folds": [float(r) for r in no_drug_folds],
        },
        "morgan_fp": {
            "mean": fp_mean,
            "std": fp_std,
            "folds": [float(r) for r in morgan_fp_folds],
            "delta": delta,
        },
        "n_scaffolds": n_scaffolds,
        "fold_sizes": fold_sizes,
        "scaffold_leak_assertion": "passed",
    }
    out_path = report_dir / "metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
