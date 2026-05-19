"""03 — K-Shot Response Matching: External Replication.

Tests whether K-shot response matching K-curve (lift + K=1 dip) replicates
in CTRPv2, BeatAML, and PRISM. Fixed blend weight w=0.5. ~1 hr with vectorized
response_match_predict.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
from src.data.prism import load_prism, preprocess_prism
from src.evaluation.per_drug import per_drug_r
from src.utils.ridge import safe_fit_scaler
from src.utils.solutions import response_match_predict

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

RNA_DIM, MUT_DIM = 550, 200
BEATAML_RNA_DIM = 500
ALPHA = 1.0
RANDOM_STATE = 42
K_VALUES = [0, 1, 3, 5, 10, 20, 50]
BLEND_W = 0.5
N_REPEATS = 20  # random anchor draws per drug per K (K>0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core K-shot evaluation
# ---------------------------------------------------------------------------

def kshot_cv(
    X_cells: np.ndarray,          # (n_cells, n_feat) — one row per unique cell
    resp_mat: np.ndarray,          # (n_drugs, n_cells) — NaN for missing
    all_drugs: list[str],
    n_folds: int,
    k_values: list[int] = K_VALUES,
    blend_w: float = BLEND_W,
    min_obs: int = 10,
    n_repeats: int = N_REPEATS,
    rng_seed: int = RANDOM_STATE,
    dataset_name: str = "",
) -> list[dict]:
    """Drug-blind K-fold CV K-curve.

    For K=0: pure Ridge.
    For K>0: blend Ridge + response_match_predict on held-out cells,
    averaged over n_repeats random anchor draws.
    """
    rng = np.random.default_rng(rng_seed)
    n_drugs = len(all_drugs)
    unique_drugs = np.arange(n_drugs)
    kf = KFold(n_splits=min(n_folds, n_drugs), shuffle=True, random_state=RANDOM_STATE)

    # Expand: for each (drug, cell) pair with valid response → training sample
    drug_idx_all, cell_idx_all, y_all = [], [], []
    for di in range(n_drugs):
        for ci in range(X_cells.shape[0]):
            v = resp_mat[di, ci]
            if not np.isnan(v):
                drug_idx_all.append(di)
                cell_idx_all.append(ci)
                y_all.append(v)
    drug_idx_all = np.array(drug_idx_all)
    cell_idx_all = np.array(cell_idx_all)
    y_all = np.array(y_all, dtype=np.float32)

    k_preds: dict[int, tuple[list, list, list]] = {k: ([], [], []) for k in k_values}
    n_folds_actual = min(n_folds, n_drugs)

    for fold_i, (fold_train_d, fold_test_d) in enumerate(kf.split(unique_drugs)):
        t_fold_start = time.time()
        train_d = set(fold_train_d)
        test_d = set(fold_test_d)

        train_mask = np.array([di in train_d for di in drug_idx_all])
        X_train = X_cells[cell_idx_all[train_mask]]
        y_train = y_all[train_mask]

        sc = safe_fit_scaler(X_train)
        model = Ridge(alpha=ALPHA)
        model.fit(sc.transform(X_train), y_train)

        ridge_all_cells = model.predict(sc.transform(X_cells))

        train_drug_idx_arr = np.array(sorted(train_d))
        train_resp = resp_mat[train_drug_idx_arr]
        cell_mean = np.nanmean(train_resp, axis=0)
        # Fill cells with no training observations (NaN) with global mean
        # to prevent NaN propagation in sparse datasets (e.g. BeatAML).
        _global = np.nanmean(cell_mean)
        cell_mean = np.where(np.isnan(cell_mean), _global, cell_mean)

        n_test = len(fold_test_d)
        for drug_i, di in enumerate(fold_test_d):
            obs_cells = np.where(~np.isnan(resp_mat[di]))[0]
            if len(obs_cells) < min_obs + 1:
                continue
            drug_y = resp_mat[di, obs_cells]
            ridge_pred = ridge_all_cells[obs_cells]
            drug_name = all_drugs[di]

            for k in k_values:
                if k == 0:
                    k_preds[k][0].append(ridge_pred)
                    k_preds[k][1].append(drug_y)
                    k_preds[k][2].append(np.full(len(ridge_pred), drug_name))
                    continue

                k_actual = min(k, len(obs_cells) - min_obs)
                if k_actual <= 0:
                    continue

                for _ in range(n_repeats):
                    anchor_pos = rng.choice(len(obs_cells), size=k_actual, replace=False)
                    eval_pos = np.setdiff1d(np.arange(len(obs_cells)), anchor_pos)
                    if len(eval_pos) < min_obs:
                        continue

                    anchor_cells = obs_cells[anchor_pos]
                    obs_vals = drug_y[anchor_pos]

                    match_full = response_match_predict(
                        train_resp, obs_vals, anchor_cells, cell_mean, blend_weight=1.0,
                    )
                    blended = blend_w * match_full[obs_cells[eval_pos]] + (1 - blend_w) * ridge_pred[eval_pos]

                    k_preds[k][0].append(blended)
                    k_preds[k][1].append(drug_y[eval_pos])
                    k_preds[k][2].append(np.full(len(blended), drug_name))

            if (drug_i + 1) % 20 == 0:
                log.info("  %s fold %d/%d: %d/%d test drugs done", dataset_name,
                         fold_i + 1, n_folds_actual, drug_i + 1, n_test)

        elapsed = time.time() - t_fold_start
        log.info("  %s fold %d/%d done in %.1fs (%d test drugs)", dataset_name,
                 fold_i + 1, n_folds_actual, elapsed, n_test)

    results = []
    for k in k_values:
        preds, targets, names = k_preds[k]
        if not preds:
            results.append({"k": k, "per_drug_r": float("nan")})
            continue
        rs = per_drug_r(np.concatenate(preds), np.concatenate(targets),
                        np.concatenate(names), min_cells=min_obs)
        pdr = float(np.mean(list(rs.values()))) if rs else float("nan")
        results.append({"k": k, "per_drug_r": pdr})
        log.info("  K=%2d: per_drug_r=%.4f", k, pdr)
    return results


def build_resp_matrix_generic(
    df: pd.DataFrame,
    all_drugs: list[str],
    all_cells: list[str],
    drug_col: str,
    cell_col: str,
    response_col: str,
) -> np.ndarray:
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    mat = np.full((len(all_drugs), len(all_cells)), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        d, c = row[drug_col], row[cell_col]
        if d in drug_to_idx and c in cell_to_idx:
            mat[drug_to_idx[d], cell_to_idx[c]] = row[response_col]
    return mat


def pca_cell_features(arr: np.ndarray, n: int) -> np.ndarray:
    n = min(n, arr.shape[0] - 1, arr.shape[1])
    return PCA(n_components=n, random_state=RANDOM_STATE).fit_transform(
        arr.astype(np.float64)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset runners
# ---------------------------------------------------------------------------

def run_ctrpv2(smoke: bool = False) -> dict:
    log.info("=== CTRPv2 ===")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mut = pd.read_parquet(DATA_DIR / "mutations.parquet")

    df = load_ctrpv2_response()
    df = filter_ctrpv2(df, rna.index, mut.index, min_cells=10)

    all_cells = sorted(df["depmap_id"].unique())
    all_drugs = sorted(df["drug_name"].unique())

    if smoke:
        all_drugs = all_drugs[:20]
        df = df[df["drug_name"].isin(all_drugs)].copy()

    log.info("CTRPv2: %d drugs, %d cells", len(all_drugs), len(all_cells))

    rna_pca = pca_cell_features(rna.loc[all_cells].values.astype(np.float32), RNA_DIM)
    mut_pca = pca_cell_features(mut.loc[all_cells].values.astype(np.float32), MUT_DIM)
    X_cells = np.concatenate([rna_pca, mut_pca], axis=1)

    resp_mat = build_resp_matrix_generic(df, all_drugs, all_cells, "drug_name", "depmap_id", "auc")

    k_vals = [0, 1, 50] if smoke else K_VALUES
    n_rep = 2 if smoke else N_REPEATS
    k_curve = kshot_cv(X_cells, resp_mat, all_drugs, n_folds=2 if smoke else 5,
                       k_values=k_vals, n_repeats=n_rep, min_obs=10, dataset_name="CTRPv2")
    return {"k_curve": k_curve, "n_drugs": len(all_drugs), "n_cells": len(all_cells)}


def run_beataml(smoke: bool = False) -> dict:
    log.info("=== BeatAML ===")
    response = load_beataml_response(min_patients=20)
    patients = sorted(response["patient_id"].unique())
    drugs = sorted(response["drug"].unique())

    if smoke:
        drugs = drugs[:20]
        response = response[response["drug"].isin(drugs)].copy()

    expr = load_beataml_expression(patients, top_genes=5000)
    common = sorted(set(patients) & set(expr.index))
    response = response[response["patient_id"].isin(common)].copy()
    drugs = sorted(response["drug"].unique())

    log.info("BeatAML: %d drugs, %d patients", len(drugs), len(common))

    sc_std = StandardScaler()
    X_cells = PCA(
        n_components=min(BEATAML_RNA_DIM, len(common) - 1), random_state=RANDOM_STATE
    ).fit_transform(sc_std.fit_transform(expr.loc[common].values)).astype(np.float32)

    resp_mat = build_resp_matrix_generic(response, drugs, common, "drug", "patient_id", "auc")

    k_vals = [0, 1, 50] if smoke else K_VALUES
    n_rep = 2 if smoke else N_REPEATS
    k_curve = kshot_cv(X_cells, resp_mat, drugs, n_folds=2 if smoke else 5,
                       k_values=k_vals, n_repeats=n_rep, min_obs=10, dataset_name="BeatAML")
    return {
        "k_curve": k_curve,
        "n_drugs": len(drugs),
        "n_patients": len(common),
        "note": f"N_REPEATS={N_REPEATS} random anchor draws per drug per K",
    }


def run_prism(smoke: bool = False) -> dict:
    log.info("=== PRISM ===")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mut = pd.read_parquet(DATA_DIR / "mutations.parquet")

    df, n_drugs, n_cells = preprocess_prism(
        load_prism(DATA_DIR), set(rna.index) & set(mut.index), min_cells_per_drug=50,
    )
    all_cells = sorted(df["depmap_id"].unique())
    all_drugs = sorted(df["drug_name"].unique())

    if smoke:
        all_drugs = all_drugs[:30]
        df = df[df["drug_name"].isin(all_drugs)].copy()
        n_drugs = len(all_drugs)

    log.info("PRISM: %d drugs, %d cells", n_drugs, n_cells)

    rna_pca = pca_cell_features(rna.loc[all_cells].values.astype(np.float32), RNA_DIM)
    mut_pca = pca_cell_features(mut.loc[all_cells].values.astype(np.float32), MUT_DIM)
    X_cells = np.concatenate([rna_pca, mut_pca], axis=1)

    resp_mat = build_resp_matrix_generic(df, all_drugs, all_cells, "drug_name", "depmap_id", "response")

    k_vals = [0, 1, 50] if smoke else K_VALUES
    n_rep = 2 if smoke else N_REPEATS
    k_curve = kshot_cv(X_cells, resp_mat, all_drugs, n_folds=2 if smoke else 10,
                       k_values=k_vals, n_repeats=n_rep, min_obs=50, dataset_name="PRISM")
    note = "Same cell lines as GDSC2 omics — tests drug panel breadth (1415 drugs), not cell OOD transfer"
    return {"k_curve": k_curve, "n_drugs": n_drugs, "n_cells": n_cells, "note": note}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Small subset + fewer K values for quick check")
    parser.add_argument("--datasets", nargs="+", choices=["ctrpv2", "beataml", "prism"],
                        default=["ctrpv2", "beataml", "prism"], help="Datasets to run")
    args = parser.parse_args()

    if args.smoke:
        log.info("SMOKE MODE: small drug subsets, K=[0,1,50], n_repeats=2, n_folds=2")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXP_DIR / "results" / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runners = {"ctrpv2": run_ctrpv2, "beataml": run_beataml, "prism": run_prism}
    results: dict = {}
    for ds in args.datasets:
        results[ds] = runners[ds](smoke=args.smoke)

    if len(args.datasets) == 3:
        gates = []
        for ds_name, ds in results.items():
            curve = {e["k"]: e["per_drug_r"] for e in ds["k_curve"]}
            k0 = curve.get(0, float("nan"))
            k50 = curve.get(50, float("nan"))
            k1 = curve.get(1, float("nan"))
            gates.append({
                "dataset": ds_name,
                "k50_gt_k0": bool(k50 > k0) if not (np.isnan(k50) or np.isnan(k0)) else None,
                "k1_lt_k0": bool(k1 < k0) if not (np.isnan(k1) or np.isnan(k0)) else None,
            })
        results["gates"] = gates

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
