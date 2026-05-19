"""01 — Drug Feature Null: External Replication.

Tests whether Morgan FP Δ ≈ 0 for per-drug r in CTRPv2, BeatAML, and PRISM.
All Ridge, all CPU. Expected runtime ~30 min.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.data.beataml import load_beataml_response, load_beataml_expression
from src.data.ctrpv2 import load_ctrpv2_response, filter_ctrpv2
from src.data.drug_features import compute_fingerprints, fetch_smiles
from src.data.prism import load_prism, preprocess_prism
from src.evaluation.per_drug import per_drug_r
from src.utils.ridge import safe_fit_scaler

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
SMILES_CACHE = DATA_DIR / "drug_smiles.json"

RNA_DIM, MUT_DIM = 550, 200
BEATAML_RNA_DIM = 500
ALPHA = 1.0
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature utilities
# ---------------------------------------------------------------------------

def pca_compress(arr: np.ndarray, train_rows: np.ndarray, n_components: int) -> np.ndarray:
    n = min(n_components, len(np.unique(train_rows)) - 1, arr.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    pca.fit(arr[train_rows].astype(np.float64))
    return pca.transform(arr.astype(np.float64)).astype(np.float32)


def get_morgan_fp(drug_names: list[str]) -> np.ndarray:
    drug_to_idx = {d: i for i, d in enumerate(drug_names)}
    smiles = fetch_smiles(drug_names, SMILES_CACHE)
    return compute_fingerprints(smiles, drug_to_idx)


def run_drug_blind_cv(
    X_cell: np.ndarray,
    y: np.ndarray,
    drug_idx: np.ndarray,
    drug_names: np.ndarray,
    drug_fp: np.ndarray | None,
    n_folds: int,
    min_cells: int = 5,
    dataset_name: str = "",
    condition: str = "",
) -> float:
    """Drug-blind CV returning mean per-drug r."""
    unique_drugs = np.unique(drug_idx)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    all_preds, all_targets, all_drug_names = [], [], []

    for fold_i, (train_d, test_d) in enumerate(kf.split(unique_drugs)):
        train_drugs = set(unique_drugs[train_d])
        test_drugs = set(unique_drugs[test_d])
        train_mask = np.array([d in train_drugs for d in drug_idx])
        test_mask = np.array([d in test_drugs for d in drug_idx])

        X_train_cell = X_cell[train_mask]
        X_test_cell = X_cell[test_mask]

        if drug_fp is not None:
            X_train = np.concatenate([X_train_cell, drug_fp[drug_idx[train_mask]]], axis=1)
            X_test = np.concatenate([X_test_cell, drug_fp[drug_idx[test_mask]]], axis=1)
        else:
            X_train, X_test = X_train_cell, X_test_cell

        sc = safe_fit_scaler(X_train)
        model = Ridge(alpha=ALPHA)
        model.fit(sc.transform(X_train), y[train_mask])
        preds = model.predict(sc.transform(X_test))

        all_preds.append(preds)
        all_targets.append(y[test_mask])
        all_drug_names.append(drug_names[test_mask])
        log.info("  %s [%s] fold %d/%d done (%d test drugs)", dataset_name, condition,
                 fold_i + 1, n_folds, len(test_d))

    preds_all = np.concatenate(all_preds)
    targets_all = np.concatenate(all_targets)
    drugs_all = np.concatenate(all_drug_names)
    rs = per_drug_r(preds_all, targets_all, drugs_all, min_cells=min_cells)
    return float(np.mean(list(rs.values()))) if rs else float("nan")


# ---------------------------------------------------------------------------
# Dataset runners
# ---------------------------------------------------------------------------

def run_ctrpv2(smoke: bool = False) -> dict:
    log.info("=== CTRPv2 ===")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mut = pd.read_parquet(DATA_DIR / "mutations.parquet")

    df = load_ctrpv2_response()
    df = filter_ctrpv2(df, rna.index, mut.index, min_cells=10)
    if smoke:
        drugs_subset = sorted(df["drug_name"].unique())[:30]
        df = df[df["drug_name"].isin(drugs_subset)].copy()

    all_cells = sorted(df["depmap_id"].unique())
    all_drugs = sorted(df["drug_name"].unique())
    n_drugs = len(all_drugs)
    log.info("CTRPv2: %d drugs, %d cells", n_drugs, len(all_cells))

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mut.loc[all_cells].values.astype(np.float32)
    cell_to_row = {c: i for i, c in enumerate(all_cells)}
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}

    rows = [cell_to_row[r["depmap_id"]] for _, r in df.iterrows()]
    drug_idx = np.array([drug_to_idx[r["drug_name"]] for _, r in df.iterrows()])
    drug_names = np.array([r["drug_name"] for _, r in df.iterrows()])
    y = df["auc"].values.astype(np.float32)

    train_rows = np.arange(len(all_cells))
    rna_pca = pca_compress(rna_arr, train_rows, RNA_DIM)
    mut_pca = pca_compress(mut_arr, train_rows, MUT_DIM)
    X_cell = np.concatenate([rna_pca[rows], mut_pca[rows]], axis=1)

    fp = get_morgan_fp(all_drugs)

    n_folds = min(5 if smoke else 10, n_drugs)
    no_drug_r = run_drug_blind_cv(X_cell, y, drug_idx, drug_names, None, n_folds,
                                   dataset_name="CTRPv2", condition="no_drug")
    morgan_r = run_drug_blind_cv(X_cell, y, drug_idx, drug_names, fp, n_folds,
                                  dataset_name="CTRPv2", condition="morgan_fp")
    delta = morgan_r - no_drug_r
    log.info("CTRPv2: no_drug=%.4f  morgan=%.4f  Δ=%.4f", no_drug_r, morgan_r, delta)
    return {"no_drug_r": no_drug_r, "morgan_fp_r": morgan_r, "delta": delta, "n_drugs": n_drugs}


def run_beataml(smoke: bool = False) -> dict:
    log.info("=== BeatAML ===")
    response = load_beataml_response(min_patients=20)
    patients = sorted(response["patient_id"].unique())
    drugs = sorted(response["drug"].unique())

    if smoke:
        drugs = drugs[:20]
        response = response[response["drug"].isin(drugs)].copy()

    n_drugs = len(drugs)
    expr = load_beataml_expression(patients, top_genes=5000)
    common = sorted(set(patients) & set(expr.index))
    response = response[response["patient_id"].isin(common)].copy()
    log.info("BeatAML: %d drugs, %d patients", n_drugs, len(common))

    sc = StandardScaler()
    expr_scaled = sc.fit_transform(expr.loc[common].values)
    pca = PCA(n_components=min(BEATAML_RNA_DIM, expr_scaled.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(expr_scaled).astype(np.float32)
    patient_to_row = {p: i for i, p in enumerate(common)}
    drug_to_idx = {d: i for i, d in enumerate(drugs)}

    rows = [patient_to_row[r["patient_id"]] for _, r in response.iterrows()]
    drug_idx = np.array([drug_to_idx[r["drug"]] for _, r in response.iterrows()])
    drug_names = np.array([r["drug"] for _, r in response.iterrows()])
    y = response["auc"].values.astype(np.float32)
    X_cell = X_pca[rows]

    fp = get_morgan_fp(drugs)

    n_folds = min(3 if smoke else 5, n_drugs)
    no_drug_r = run_drug_blind_cv(X_cell, y, drug_idx, drug_names, None, n_folds,
                                   dataset_name="BeatAML", condition="no_drug")
    morgan_r = run_drug_blind_cv(X_cell, y, drug_idx, drug_names, fp, n_folds,
                                  dataset_name="BeatAML", condition="morgan_fp")
    delta = morgan_r - no_drug_r
    log.info("BeatAML: no_drug=%.4f  morgan=%.4f  Δ=%.4f", no_drug_r, morgan_r, delta)
    return {"no_drug_r": no_drug_r, "morgan_fp_r": morgan_r, "delta": delta, "n_drugs": n_drugs}


def run_prism(smoke: bool = False) -> dict:
    log.info("=== PRISM ===")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mut = pd.read_parquet(DATA_DIR / "mutations.parquet")

    prism_raw = load_prism(DATA_DIR)
    df, n_drugs, n_cells = preprocess_prism(prism_raw, set(rna.index) & set(mut.index), min_cells_per_drug=50)

    if smoke:
        drugs_subset = sorted(df["drug_name"].unique())[:30]
        df = df[df["drug_name"].isin(drugs_subset)].copy()
        n_drugs = len(drugs_subset)

    all_cells = sorted(df["depmap_id"].unique())
    all_drugs = sorted(df["drug_name"].unique())
    log.info("PRISM: %d drugs, %d cells", n_drugs, n_cells)

    rna_arr = rna.loc[all_cells].values.astype(np.float32)
    mut_arr = mut.loc[all_cells].values.astype(np.float32)
    cell_to_row = {c: i for i, c in enumerate(all_cells)}
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}

    train_rows = np.arange(len(all_cells))
    rna_pca = pca_compress(rna_arr, train_rows, RNA_DIM)
    mut_pca = pca_compress(mut_arr, train_rows, MUT_DIM)

    rows = [cell_to_row[r["depmap_id"]] for _, r in df.iterrows()]
    drug_idx_arr = np.array([drug_to_idx[r["drug_name"]] for _, r in df.iterrows()])
    drug_names = np.array([r["drug_name"] for _, r in df.iterrows()])
    y = df["response"].values.astype(np.float32)
    X_cell = np.concatenate([rna_pca[rows], mut_pca[rows]], axis=1)

    fp = get_morgan_fp(all_drugs)

    n_folds = min(3 if smoke else 10, len(all_drugs))
    no_drug_r = run_drug_blind_cv(X_cell, y, drug_idx_arr, drug_names, None, n_folds, min_cells=50,
                                   dataset_name="PRISM", condition="no_drug")
    morgan_r = run_drug_blind_cv(X_cell, y, drug_idx_arr, drug_names, fp, n_folds, min_cells=50,
                                  dataset_name="PRISM", condition="morgan_fp")
    delta = morgan_r - no_drug_r
    log.info("PRISM: no_drug=%.4f  morgan=%.4f  Δ=%.4f", no_drug_r, morgan_r, delta)
    return {"no_drug_r": no_drug_r, "morgan_fp_r": morgan_r, "delta": delta, "n_drugs": n_drugs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (small subset)")
    parser.add_argument("--datasets", nargs="+", choices=["ctrpv2", "beataml", "prism"],
                        default=["ctrpv2", "beataml", "prism"], help="Datasets to run")
    args = parser.parse_args()

    if args.smoke:
        log.info("SMOKE MODE: small subsets, fewer folds")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXP_DIR / "results" / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runners = {"ctrpv2": run_ctrpv2, "beataml": run_beataml, "prism": run_prism}
    results: dict = {}
    for ds in args.datasets:
        results[ds] = runners[ds](smoke=args.smoke)

    if len(args.datasets) == 3:
        gate_pass = all(abs(v["delta"]) < 0.01 for v in results.values())
        results["gate"] = {"pass": gate_pass, "criterion": "|delta| < 0.01 in all datasets"}
        log.info("Gate: %s", "PASS" if gate_pass else "FAIL — check deltas")

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
