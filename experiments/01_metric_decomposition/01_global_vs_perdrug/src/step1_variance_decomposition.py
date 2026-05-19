"""Step 1: Variance decomposition of GDSC2 IC50.

Answers: what fraction of IC50 variance is between-drug vs within-drug?
The square root of the between-drug fraction is the theoretical upper bound
on global Pearson r achievable by a drug-mean oracle — without any knowledge
of cell sensitivity differences.

Runtime: < 1 minute, no GPU needed.
Output:  results/variance_decomposition.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[4]
EXP_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = EXP_DIR / "results"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    response = pd.read_parquet(DATA_DIR / "drug_response.parquet")
    rna_idx = pd.read_parquet(DATA_DIR / "rna.parquet").index
    mut_idx = pd.read_parquet(DATA_DIR / "mutations.parquet").index
    valid_cells = set(rna_idx) & set(mut_idx)
    df = response[response["depmap_id"].isin(valid_cells)].copy()
    log.info("Pairs: %d  |  Drugs: %d  |  Cells: %d",
             len(df), df["drug_name"].nunique(), df["depmap_id"].nunique())

    y = df["ln_ic50"].to_numpy(dtype=np.float64)
    total_var = float(np.var(y, ddof=1))
    log.info("Total IC50 variance: %.4f  (std=%.4f)", total_var, np.sqrt(total_var))

    # Between-drug variance = Var(E[y|d])
    drug_means = df.groupby("drug_name")["ln_ic50"].mean()
    between_drug_var = float(np.var(drug_means.values, ddof=1))

    # Within-drug variance = E[Var(y|d)]
    drug_vars = df.groupby("drug_name")["ln_ic50"].var(ddof=1).dropna()
    within_drug_var = float(drug_vars.mean())

    between_frac = between_drug_var / total_var
    within_frac = within_drug_var / total_var
    theoretical_ceiling = float(np.sqrt(between_frac))

    log.info("Between-drug variance: %.4f  (%.1f%% of total)", between_drug_var, between_frac * 100)
    log.info("Within-drug variance:  %.4f  (%.1f%% of total)", within_drug_var, within_frac * 100)
    log.info("Theoretical global-r ceiling (drug-mean oracle): %.4f", theoretical_ceiling)

    # Per-drug stats
    per_drug_stats = []
    for drug, grp in df.groupby("drug_name"):
        v = grp["ln_ic50"].to_numpy()
        per_drug_stats.append({
            "drug": drug,
            "n_cells": int(len(v)),
            "mean": round(float(np.mean(v)), 4),
            "std": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0,
        })
    per_drug_stats.sort(key=lambda x: x["mean"])

    # IC50 range stats
    ic50_range = {
        "min": round(float(y.min()), 4),
        "max": round(float(y.max()), 4),
        "mean": round(float(y.mean()), 4),
        "std": round(float(y.std(ddof=1)), 4),
    }

    result = {
        "n_pairs": int(len(df)),
        "n_drugs": int(df["drug_name"].nunique()),
        "n_cells": int(df["depmap_id"].nunique()),
        "ic50_range": ic50_range,
        "total_var": round(total_var, 6),
        "between_drug_var": round(between_drug_var, 6),
        "within_drug_var": round(within_drug_var, 6),
        "between_drug_var_fraction": round(between_frac, 6),
        "within_drug_var_fraction": round(within_frac, 6),
        "global_r_ceiling_from_between": round(theoretical_ceiling, 6),
        "interpretation": (
            f"A drug-mean oracle that knows per-drug mean IC50 but nothing about "
            f"cell sensitivity can achieve global Pearson r ≤ {theoretical_ceiling:.3f}. "
            f"Between-drug variance accounts for {between_frac*100:.1f}% of total IC50 variance."
        ),
        "per_drug_stats": per_drug_stats,
    }

    out_path = RESULTS_DIR / "variance_decomposition.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved → %s", out_path)

    # Decision criterion check (pre-registered threshold: > 50%)
    decision = "PASS" if between_frac > 0.50 else "FAIL"
    log.info("Pre-registered criterion (between_frac > 0.50): %s  (value=%.3f)",
             decision, between_frac)


if __name__ == "__main__":
    main()
