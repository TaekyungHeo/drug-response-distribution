"""02 — Within-MoA Training: External Replication.

Tests whether within-MoA LOO improves per-drug r in CTRPv2, BeatAML, and PRISM.
MoA annotations from Drug Repurposing Hub. All Ridge, all CPU. ~1 hr.

CV design: LOO for both all-drug baseline and within-MoA. Using LOO (not k-fold)
for the baseline ensures both conditions are evaluated on the same test samples per
drug, making the Δ a clean within-drug comparison.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.data.beataml import load_beataml_response, load_beataml_expression
from src.data.ctrpv2 import load_ctrpv2_response, filter_ctrpv2
from src.data.prism import load_prism, preprocess_prism
from src.data.repurposing_hub import build_drug_moa_map, group_by_moa
from src.evaluation.per_drug import per_drug_r
from src.utils.ridge import safe_fit_scaler

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

RNA_DIM, MUT_DIM = 550, 200
BEATAML_RNA_DIM = 500
ALPHA = 1.0
RANDOM_STATE = 42
MIN_DRUGS_PER_MOA = 3
MIN_CELLS_PER_DRUG = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature utilities
# ---------------------------------------------------------------------------

def pca_compress(arr: np.ndarray, n_components: int) -> np.ndarray:
    n = min(n_components, arr.shape[0] - 1, arr.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    return pca.fit_transform(arr.astype(np.float64)).astype(np.float32)


def ldo_cv(
    X_cell: np.ndarray,
    y: np.ndarray,
    drug_names: np.ndarray,
    train_drug_set: list[str] | None,
    eval_drug_set: list[str],
    min_cells: int = MIN_CELLS_PER_DRUG,
) -> dict[str, float]:
    """Leave-one-drug-out CV over eval_drug_set.

    If train_drug_set is None, trains on all drugs except the held-out drug
    (all-drug baseline). Otherwise trains only on train_drug_set minus the
    held-out drug (within-MoA).
    """
    all_preds, all_targets, all_names = [], [], []

    for held_out in eval_drug_set:
        if train_drug_set is None:
            train_mask = drug_names != held_out
        else:
            train_mask = np.isin(drug_names, train_drug_set) & (drug_names != held_out)
        test_mask = drug_names == held_out

        if train_mask.sum() < 5 or test_mask.sum() < min_cells:
            continue

        sc = safe_fit_scaler(X_cell[train_mask])
        model = Ridge(alpha=ALPHA)
        model.fit(sc.transform(X_cell[train_mask]), y[train_mask])
        preds = model.predict(sc.transform(X_cell[test_mask]))

        all_preds.append(preds)
        all_targets.append(y[test_mask])
        all_names.append(drug_names[test_mask])

    if not all_preds:
        return {}
    return per_drug_r(
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_names),
        min_cells=min_cells,
    )


def analyze_dataset(
    X_cell: np.ndarray,
    y: np.ndarray,
    drug_names: np.ndarray,
    all_drugs: list[str],
    dataset_name: str,
    min_cells: int = MIN_CELLS_PER_DRUG,
    smoke: bool = False,
) -> dict:
    drug_moa = build_drug_moa_map(all_drugs)
    moa_groups = group_by_moa(drug_moa, min_drugs=MIN_DRUGS_PER_MOA)
    log.info("%s: %d drugs with MoA, %d MoA classes", dataset_name,
             sum(v is not None for v in drug_moa.values()), len(moa_groups))

    # In smoke mode, only evaluate focus MoAs to save time
    focus_only = smoke
    focus_moas = {"EGFR inhibitor", "MEK inhibitor"}

    moa_results = []
    moa_items = sorted(moa_groups.items())
    if focus_only:
        moa_items = [(m, d) for m, d in moa_items if m in focus_moas]
        log.info("  smoke mode: evaluating only focus MoAs %s", sorted(focus_moas))

    for moa_idx, (moa, drugs_in_moa) in enumerate(moa_items):
        drugs_present = [d for d in drugs_in_moa if d in set(all_drugs)]
        if len(drugs_present) < MIN_DRUGS_PER_MOA:
            continue

        log.info("  [%d/%d] MoA=%s  n=%d", moa_idx + 1, len(moa_items), moa, len(drugs_present))

        # All-drug LOO baseline (train on all drugs except held-out)
        all_drug_rs = ldo_cv(X_cell, y, drug_names, None, drugs_present, min_cells)
        all_drug_r = float(np.mean(list(all_drug_rs.values()))) if all_drug_rs else float("nan")

        # Within-MoA LOO (train only on other drugs in same MoA)
        within_moa_rs = ldo_cv(X_cell, y, drug_names, drugs_present, drugs_present, min_cells)
        within_moa_r = float(np.mean(list(within_moa_rs.values()))) if within_moa_rs else float("nan")

        delta = within_moa_r - all_drug_r
        log.info("    all_drug=%.4f  within_moa=%.4f  Δ=%+.4f", all_drug_r, within_moa_r, delta)

        moa_results.append({
            "moa": moa,
            "n_drugs": len(drugs_present),
            "all_drug_r": all_drug_r,
            "within_moa_r": within_moa_r,
            "delta": delta,
        })

    return {
        "n_drugs_total": len(all_drugs),
        "n_drugs_with_moa": sum(1 for v in drug_moa.values() if v is not None),
        "n_moa_classes": len(moa_groups),
        "moa_classes": moa_results,
    }


# ---------------------------------------------------------------------------
# Dataset runners
# ---------------------------------------------------------------------------

def run_ctrpv2(smoke: bool = False) -> dict:
    log.info("=== CTRPv2 ===")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mut = pd.read_parquet(DATA_DIR / "mutations.parquet")

    df = load_ctrpv2_response()
    df = filter_ctrpv2(df, rna.index, mut.index, min_cells=MIN_CELLS_PER_DRUG)

    all_cells = sorted(df["depmap_id"].unique())
    all_drugs = sorted(df["drug_name"].unique())
    log.info("CTRPv2: %d drugs, %d cells", len(all_drugs), len(all_cells))
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_pca = pca_compress(rna.loc[all_cells].values.astype(np.float32), RNA_DIM)
    mut_pca = pca_compress(mut.loc[all_cells].values.astype(np.float32), MUT_DIM)

    rows = [cell_to_row[r["depmap_id"]] for _, r in df.iterrows()]
    drug_names = np.array([r["drug_name"] for _, r in df.iterrows()])
    y = df["auc"].values.astype(np.float32)
    X_cell = np.concatenate([rna_pca[rows], mut_pca[rows]], axis=1)

    return analyze_dataset(X_cell, y, drug_names, all_drugs, "CTRPv2", smoke=smoke)


def run_beataml(smoke: bool = False) -> dict:
    log.info("=== BeatAML ===")
    response = load_beataml_response(min_patients=20)
    patients = sorted(response["patient_id"].unique())
    drugs = sorted(response["drug"].unique())

    expr = load_beataml_expression(patients, top_genes=5000)
    common = sorted(set(patients) & set(expr.index))
    response = response[response["patient_id"].isin(common)].copy()
    drugs = sorted(response["drug"].unique())
    log.info("BeatAML: %d drugs, %d patients", len(drugs), len(common))

    sc = StandardScaler()
    X_pca = PCA(n_components=min(BEATAML_RNA_DIM, len(common) - 1), random_state=RANDOM_STATE).fit_transform(
        sc.fit_transform(expr.loc[common].values)
    ).astype(np.float32)
    patient_to_row = {p: i for i, p in enumerate(common)}

    rows = [patient_to_row[r["patient_id"]] for _, r in response.iterrows()]
    drug_names = np.array([r["drug"] for _, r in response.iterrows()])
    y = response["auc"].values.astype(np.float32)
    X_cell = X_pca[rows]

    return analyze_dataset(X_cell, y, drug_names, drugs, "BeatAML", min_cells=10, smoke=smoke)


def run_prism(smoke: bool = False) -> dict:
    log.info("=== PRISM ===")
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mut = pd.read_parquet(DATA_DIR / "mutations.parquet")

    df, n_drugs, n_cells = preprocess_prism(
        load_prism(DATA_DIR), set(rna.index) & set(mut.index), min_cells_per_drug=50,
    )
    all_cells = sorted(df["depmap_id"].unique())
    all_drugs = sorted(df["drug_name"].unique())
    log.info("PRISM: %d drugs, %d cells", n_drugs, n_cells)
    cell_to_row = {c: i for i, c in enumerate(all_cells)}

    rna_pca = pca_compress(rna.loc[all_cells].values.astype(np.float32), RNA_DIM)
    mut_pca = pca_compress(mut.loc[all_cells].values.astype(np.float32), MUT_DIM)

    rows = [cell_to_row[r["depmap_id"]] for _, r in df.iterrows()]
    drug_names = np.array([r["drug_name"] for _, r in df.iterrows()])
    y = df["response"].values.astype(np.float32)
    X_cell = np.concatenate([rna_pca[rows], mut_pca[rows]], axis=1)

    return analyze_dataset(X_cell, y, drug_names, all_drugs, "PRISM", min_cells=50, smoke=smoke)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Evaluate only focus MoAs for speed")
    parser.add_argument("--datasets", nargs="+", choices=["ctrpv2", "beataml", "prism"],
                        default=["ctrpv2", "beataml", "prism"], help="Datasets to run")
    args = parser.parse_args()

    if args.smoke:
        log.info("SMOKE MODE: focus MoAs only (EGFR inhibitor, MEK inhibitor)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXP_DIR / "results" / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runners = {"ctrpv2": run_ctrpv2, "beataml": run_beataml, "prism": run_prism}
    datasets: dict = {}
    for ds in args.datasets:
        datasets[ds] = runners[ds](smoke=args.smoke)

    results: dict = {"datasets": datasets}

    # Gate: EGFR and MEK improve in ≥ 2/3 datasets
    focus = ["EGFR inhibitor", "MEK inhibitor"]
    gate_checks = []
    for ds_name, ds in datasets.items():
        for entry in ds["moa_classes"]:
            if entry["moa"] in focus:
                gate_checks.append({"dataset": ds_name, "moa": entry["moa"],
                                    "delta": entry["delta"], "pass": entry["delta"] > 0})

    results["gate"] = {
        "focus_moas": focus,
        "checks": gate_checks,
        "summary": f"{sum(c['pass'] for c in gate_checks)}/{len(gate_checks)} checks pass",
    }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved: %s", out_path)
    log.info("Gate: %s", results["gate"]["summary"])


if __name__ == "__main__":
    main()
