"""02_lincs_prediction: Can LINCS signatures be predicted from Morgan fingerprints?

Gate experiment.  Ridge regression: Morgan FP (2048) -> LINCS PCA(64).
Leave-one-drug-out CV on the ~104 drugs with both Morgan FP and LINCS coverage.
Also runs a permuted-FP control (shuffle drug labels on FPs).

Expected result: R^2 < 0 (structure cannot predict transcriptional effect).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.data.drug_features import get_drug_fingerprints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lincs_pca64(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load LINCS signatures and PCA to 64 dims.

    PCA is fit once on all matched drugs (not per fold) so that the 64
    components are a fixed target across all LOO iterations.

    Returns:
        (lincs_pca64, matched_drug_names) where lincs_pca64 is (n_drugs, 64).
    """
    idx_path = data_dir / "lincs_drug_index.json"
    with idx_path.open() as f:
        idx_data = json.load(f)
    matched_drugs: list[str] = idx_data["matched_drugs"]

    # Try pre-computed PCA64 first, then raw signatures
    for sig_fname, needs_pca in [("lincs_pca64.npy", False), ("lincs_signatures.npy", True)]:
        sig_path = data_dir / sig_fname
        if not sig_path.exists():
            continue
        raw = np.load(sig_path).astype(np.float64)
        # Align rows: take first raw.shape[0] drugs from matched list
        matched_drugs = matched_drugs[: raw.shape[0]]
        assert len(matched_drugs) == raw.shape[0], (
            f"Mismatch: {len(matched_drugs)} drug names vs {raw.shape[0]} signature rows"
        )
        if needs_pca:
            n_components = min(64, raw.shape[0] - 1, raw.shape[1])
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            lincs_pca = pca.fit_transform(raw)
            logger.info(
                "PCA(%d) on %s: explained variance = %.3f",
                n_components, sig_fname, pca.explained_variance_ratio_.sum(),
            )
        else:
            lincs_pca = raw
            pca = None
        logger.info("Loaded LINCS: %d drugs, %d dims", lincs_pca.shape[0], lincs_pca.shape[1])
        return lincs_pca.astype(np.float64), matched_drugs

    raise FileNotFoundError("No LINCS signature file found in " + str(data_dir))


# ---------------------------------------------------------------------------
# LOO Ridge regression
# ---------------------------------------------------------------------------

def run_loo_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    drug_names: list[str],
) -> dict:
    """Leave-one-drug-out Ridge: X -> Y.

    Returns dict with predictions, best alphas, etc.
    """
    n_drugs = X.shape[0]
    Y_pred = np.zeros_like(Y)
    best_alphas: list[float] = []

    for i in range(n_drugs):
        train_mask = np.ones(n_drugs, dtype=bool)
        train_mask[i] = False

        X_train, Y_train = X[train_mask], Y[train_mask]
        X_test = X[i : i + 1]

        ridge = RidgeCV(alphas=ALPHAS, fit_intercept=True)
        ridge.fit(X_train, Y_train)

        Y_pred[i] = ridge.predict(X_test).ravel()
        best_alphas.append(float(ridge.alpha_))

    # Per-component R^2
    per_comp_r2 = r2_score(Y, Y_pred, multioutput="raw_values")
    overall_r2 = float(per_comp_r2.mean())

    # Cosine similarity per drug
    cos_sims = np.array([
        float(cosine_similarity(Y[i : i + 1], Y_pred[i : i + 1])[0, 0])
        for i in range(n_drugs)
    ])
    mean_cosine = float(cos_sims.mean())

    # Most common alpha
    alpha_counts = Counter(best_alphas)
    best_alpha_mode = alpha_counts.most_common(1)[0][0]

    return {
        "Y_pred": Y_pred,
        "per_comp_r2": per_comp_r2,
        "overall_r2": overall_r2,
        "mean_cosine_sim": mean_cosine,
        "cos_sims": cos_sims,
        "best_alphas": best_alphas,
        "best_alpha_mode": best_alpha_mode,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test mode: subsample to 20 drugs.",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    logger.info(
        "02_lincs_prediction | started at %s | smoke=%s",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        args.smoke,
    )

    # ---- Load LINCS PCA(64) targets ----
    lincs_pca, matched_drugs = load_lincs_pca64(DATA_DIR)
    n_drugs_full = len(matched_drugs)

    # ---- Load Morgan FP for matched drugs ----
    drug_to_idx = {d: i for i, d in enumerate(matched_drugs)}
    fp_matrix = get_drug_fingerprints(drug_to_idx, DATA_DIR)
    logger.info("Morgan FP: %s", fp_matrix.shape)

    # Sanity: FP should have non-trivial variance
    fp_var = fp_matrix.var(axis=0).mean()
    logger.info("Morgan FP mean column variance: %.4f", fp_var)
    assert fp_var > 1e-6, f"FP variance too low: {fp_var}"

    # Sanity: LINCS PCA should have non-trivial variance per component
    lincs_var = lincs_pca.var(axis=0)
    logger.info(
        "LINCS PCA variance: min=%.4f, max=%.4f, mean=%.4f",
        lincs_var.min(), lincs_var.max(), lincs_var.mean(),
    )

    # ---- Smoke mode: subsample ----
    if args.smoke:
        rng = np.random.default_rng(RANDOM_SEED)
        n_smoke = min(20, n_drugs_full)
        idx = np.sort(rng.choice(n_drugs_full, size=n_smoke, replace=False))
        lincs_pca = lincs_pca[idx]
        fp_matrix = fp_matrix[idx]
        matched_drugs = [matched_drugs[i] for i in idx]
        logger.info("SMOKE: subsampled to %d drugs", n_smoke)

    n_drugs = len(matched_drugs)
    n_lincs_dim = lincs_pca.shape[1]

    X = fp_matrix.astype(np.float64)
    Y = lincs_pca.astype(np.float64)

    # ---- Real condition: LOO Ridge ----
    logger.info("Running LOO Ridge (real)...")
    real = run_loo_ridge(X, Y, matched_drugs)
    logger.info(
        "Real: overall R^2=%.4f, mean cosine=%.4f, best alpha (mode)=%.1f",
        real["overall_r2"], real["mean_cosine_sim"], real["best_alpha_mode"],
    )

    # ---- Permuted control: shuffle FP drug labels ----
    logger.info("Running LOO Ridge (permuted FP)...")
    rng = np.random.default_rng(RANDOM_SEED + 1)
    perm = rng.permutation(n_drugs)
    X_perm = X[perm]
    perm_result = run_loo_ridge(X_perm, Y, matched_drugs)
    logger.info(
        "Permuted: overall R^2=%.4f, mean cosine=%.4f",
        perm_result["overall_r2"], perm_result["mean_cosine_sim"],
    )

    # ---- Build per-component table ----
    per_component = []
    for pc in range(n_lincs_dim):
        per_component.append({
            "pc": pc,
            "r2": float(real["per_comp_r2"][pc]),
        })

    # ---- Build prediction CSV ----
    rows = []
    for i, drug in enumerate(matched_drugs):
        row = {"drug": drug}
        for pc in range(n_lincs_dim):
            row[f"true_pc{pc}"] = float(Y[i, pc])
            row[f"pred_pc{pc}"] = float(real["Y_pred"][i, pc])
        row["cosine_sim"] = float(real["cos_sims"][i])
        rows.append(row)

    # ---- Gate decision ----
    gate_pass = real["overall_r2"] <= 0.1
    logger.info(
        "Gate: R^2=%.4f %s 0.1 => %s",
        real["overall_r2"],
        "<=" if gate_pass else ">",
        "PASS (structure cannot predict LINCS)" if gate_pass else "FAIL (structure predicts LINCS)",
    )

    # ---- Write results ----
    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n_drugs": n_drugs,
        "n_drugs_full": n_drugs_full,
        "morgan_fp_dim": int(X.shape[1]),
        "lincs_pca_dim": n_lincs_dim,
        "best_alpha": real["best_alpha_mode"],
        "overall_r2": real["overall_r2"],
        "mean_cosine_sim": real["mean_cosine_sim"],
        "permuted_r2": perm_result["overall_r2"],
        "permuted_cosine_sim": perm_result["mean_cosine_sim"],
        "gate_pass": gate_pass,
        "per_component": per_component,
        "smoke": args.smoke,
        "elapsed_s": time.perf_counter() - t_start,
    }

    results_path = report_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", results_path)

    # Save prediction CSV
    import csv
    csv_path = report_dir / "prediction_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Predictions written to %s", csv_path)

    # ---- Summary ----
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  n_drugs:          %d", n_drugs)
    logger.info("  Morgan FP dim:    %d", X.shape[1])
    logger.info("  LINCS PCA dim:    %d", n_lincs_dim)
    logger.info("  Best alpha:       %.1f", real["best_alpha_mode"])
    logger.info("  Overall R^2:      %.4f", real["overall_r2"])
    logger.info("  Mean cosine sim:  %.4f", real["mean_cosine_sim"])
    logger.info("  Permuted R^2:     %.4f", perm_result["overall_r2"])
    logger.info("  Gate:             %s", "PASS" if gate_pass else "FAIL")
    logger.info("  Elapsed:          %.1f s", elapsed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
