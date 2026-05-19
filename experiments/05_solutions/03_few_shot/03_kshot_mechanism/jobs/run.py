"""Phase 34: BeatAML K=1 Mechanism — potency calibration in patient data.

Extends Phase 30 (GDSC2 K=1 mechanism) to BeatAML patient-derived data.

At K=1, compares:
  Method A: response_match — find nearest training drug by |train_drug[obs_patient] - obs_auc|
  Method B: mean_potency   — find nearest training drug by |mean(train_drug) - obs_auc|
  Method C: fp_baseline    — Morgan FP nearest drug (zero-shot)

If A ≈ B: K=1 in patient data is also purely potency calibration.
If A >> B: patient-specific response at K=1 carries profile information.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(ROOT))
from src.utils.response_matching import build_response_matrix
DATA_DIR = ROOT / "data" / "processed"
BEATAML_DIR = ROOT / "data" / "external" / "beataml2"
EXP_DIR = Path(__file__).parents[1]

N_FOLDS = 5
MIN_PATIENTS_PER_DRUG = 10  # need enough patients to evaluate K=1 then test remaining
N_REPEATS = 30
TOP_N = 5


def load_beataml():
    """Load BeatAML drug response and RNA data."""
    # Drug response
    auc_files = list(BEATAML_DIR.glob("*auc*"))
    if not auc_files:
        # Try other common names
        candidates = list(BEATAML_DIR.glob("*.csv")) + list(BEATAML_DIR.glob("*.parquet"))
        print(f"Available files: {[f.name for f in candidates[:10]]}")
        raise FileNotFoundError(f"No AUC file found in {BEATAML_DIR}")
    auc_file = auc_files[0]
    print(f"Loading: {auc_file.name}")
    if auc_file.suffix == ".parquet":
        dr = pd.read_parquet(auc_file)
    else:
        dr = pd.read_csv(auc_file)
    print(f"Columns: {dr.columns.tolist()}")
    return dr


def method_A_cell_specific(obs_auc, obs_patient_idx, train_mat, pred_patients, top_n=TOP_N):
    """Match by response at specific observed patient."""
    col = train_mat[:, obs_patient_idx]
    valid = ~np.isnan(col)
    if valid.sum() < top_n:
        return None
    dists = np.where(valid, np.abs(col - obs_auc), np.inf)
    top_idx = np.argsort(dists)[:top_n]
    weights = 1.0 / (dists[top_idx] + 1e-6)
    weights /= weights.sum()
    preds = np.zeros(len(pred_patients))
    for j, pp in enumerate(pred_patients):
        col_j = train_mat[top_idx, pp]
        v = ~np.isnan(col_j)
        if v.sum() == 0:
            preds[j] = np.nan
        else:
            w = weights[v] / weights[v].sum()
            preds[j] = np.dot(w, col_j[v])
    return preds


def method_B_mean_potency(obs_auc, train_mat, pred_patients, top_n=TOP_N):
    """Match by drug mean AUC (pure potency)."""
    means = np.nanmean(train_mat, axis=1)
    valid = ~np.isnan(means)
    if valid.sum() < top_n:
        return None
    dists = np.where(valid, np.abs(means - obs_auc), np.inf)
    top_idx = np.argsort(dists)[:top_n]
    weights = 1.0 / (dists[top_idx] + 1e-6)
    weights /= weights.sum()
    preds = np.zeros(len(pred_patients))
    for j, pp in enumerate(pred_patients):
        col_j = train_mat[top_idx, pp]
        v = ~np.isnan(col_j)
        if v.sum() == 0:
            preds[j] = np.nan
        else:
            w = weights[v] / weights[v].sum()
            preds[j] = np.dot(w, col_j[v])
    return preds


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXP_DIR / "results" / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load BeatAML
    try:
        dr = load_beataml()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        # Try to find data in alternative locations
        for p in ROOT.rglob("*beataml*"):
            print(f"  Found: {p}")
        return

    drug_col = next((c for c in dr.columns if "inhibitor" in c.lower() or "drug" in c.lower()), None)
    patient_col = next((c for c in dr.columns if "sample" in c.lower() or "patient" in c.lower()
                        or "rnaseq" in c.lower()), None)
    auc_col = next((c for c in dr.columns if "auc" in c.lower()), None)

    if not all([drug_col, patient_col, auc_col]):
        print(f"Could not identify columns. Available: {dr.columns.tolist()}")
        return

    dr_clean = dr[[drug_col, patient_col, auc_col]].dropna()
    dr_clean.columns = ["drug", "patient", "auc"]

    all_drugs = sorted(dr_clean["drug"].unique())
    all_patients = sorted(dr_clean["patient"].unique())
    print(f"BeatAML: {len(all_drugs)} drugs, {len(all_patients)} patients")

    # Build response matrix
    mat = build_response_matrix(dr_clean.rename(columns={"drug": drug_col,
                                                           "patient": patient_col,
                                                           "auc": auc_col}),
                                all_drugs, all_patients)
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    patient_to_idx = {p: i for i, p in enumerate(all_patients)}

    # Drugs with enough patients
    n_per_drug = (~np.isnan(mat)).sum(axis=1)
    eligible = [d for d in all_drugs if n_per_drug[drug_to_idx[d]] >= MIN_PATIENTS_PER_DRUG]
    print(f"Eligible drugs (≥{MIN_PATIENTS_PER_DRUG} patients): {len(eligible)}")

    # 5-fold drug-blind CV
    rng_main = np.random.default_rng(42)
    drug_perm = rng_main.permutation(eligible)
    folds = np.array_split(drug_perm, N_FOLDS)

    methods = ["A_cell_specific", "B_mean_potency"]
    all_drug_rs = {m: [] for m in methods}

    for fold_i in range(N_FOLDS):
        test_drugs = list(folds[fold_i])
        train_drugs = [d for j, f in enumerate(folds) if j != fold_i for d in f]
        train_idxs = np.array([drug_to_idx[d] for d in train_drugs])
        train_mat = mat[train_idxs]

        fold_rs = {m: [] for m in methods}
        rng_fold = np.random.default_rng(100 + fold_i)

        for test_drug in test_drugs:
            ti = drug_to_idx[test_drug]
            obs_row = mat[ti]
            valid_patients = np.where(~np.isnan(obs_row))[0]
            if len(valid_patients) < MIN_PATIENTS_PER_DRUG:
                continue

            drug_rs_reps = {m: [] for m in methods}
            for _ in range(N_REPEATS):
                chosen = rng_fold.choice(valid_patients, size=1, replace=False)
                eval_patients = np.array([p for p in valid_patients if p not in chosen])
                if len(eval_patients) < 5:
                    continue

                obs_auc = float(obs_row[chosen[0]])
                true_auc = obs_row[eval_patients]

                pA = method_A_cell_specific(obs_auc, chosen[0], train_mat, eval_patients)
                pB = method_B_mean_potency(obs_auc, train_mat, eval_patients)

                for m, preds in [("A_cell_specific", pA), ("B_mean_potency", pB)]:
                    if preds is None:
                        continue
                    valid = ~np.isnan(preds) & ~np.isnan(true_auc)
                    if valid.sum() < 5 or true_auc[valid].std() < 1e-8 or preds[valid].std() < 1e-8:
                        continue
                    r, _ = pearsonr(preds[valid], true_auc[valid])
                    drug_rs_reps[m].append(float(r))

            for m in methods:
                if drug_rs_reps[m]:
                    fold_rs[m].append(float(np.mean(drug_rs_reps[m])))

        for m in methods:
            all_drug_rs[m].extend(fold_rs[m])

        print(f"Fold {fold_i+1}/{N_FOLDS}: " +
              " | ".join(f"{m}: {np.mean(fold_rs[m]):.4f}" for m in methods if fold_rs[m]))

    print("\n=== BEATAML K=1 MECHANISM RESULTS ===")
    summary = {}
    for m in methods:
        vals = all_drug_rs[m]
        mr = float(np.mean(vals)) if vals else float("nan")
        sr = float(np.std(vals)) if vals else float("nan")
        summary[m] = {"mean": round(mr, 4), "std": round(sr, 4), "n": len(vals)}
        print(f"  {m:22s}  per-drug r = {mr:.4f} ± {sr:.4f}  (n={len(vals)})")

    if all_drug_rs["A_cell_specific"] and all_drug_rs["B_mean_potency"]:
        delta = summary["A_cell_specific"]["mean"] - summary["B_mean_potency"]["mean"]
        summary["delta_A_minus_B"] = round(delta, 4)
        print(f"\n  Δ(A - B) = {delta:+.4f}")
        if abs(delta) < 0.01:
            print("  CONCLUSION: K=1 in BeatAML is also purely potency calibration (A ≈ B)")
        else:
            print("  CONCLUSION: K=1 in BeatAML contains patient-specific profile information (A ≠ B)")

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
