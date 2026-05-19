"""Extract single-source-of-truth metrics from per-experiment results.

Reads per-experiment report/data/metrics.json and validation files,
writes a flat paper/data/metrics.json with canonical values.

Usage:
    python scripts/extract_ssot.py [--check]

    --check  Print values only, do not write (drift detection mode)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "paper" / "data" / "metrics.json"

# ── helpers ────────────────────────────────────────────────────────────────

def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def latest_run(results_dir: Path) -> Path | None:
    runs = sorted(results_dir.glob("run_*"))
    for r in reversed(runs):
        if (r / "results.json").exists():
            return r / "results.json"
    return None


# ── per-experiment adapters ────────────────────────────────────────────────

def extract_drug_feature_null() -> dict:
    """Ridge no-drug vs Morgan fingerprints from pca_ablation."""
    path = EXP / "02_drug_feature_null" / "07_pca_ablation" / "report" / "data" / "metrics.json"
    d = load(path)
    return {
        "ridgeNoDrugPerDrugR": round(d["no_drug_features"]["mean"], 4),
        "ridgeNoDrugPerDrugRStd": round(d["no_drug_features"]["std"], 4),
        "ridgeMorganPerDrugR": round(d["morgan_fp"]["mean"], 4),
        "ridgeMorganDeltaPerDrugR": round(d["morgan_fp"]["delta"], 4),
        "_src_drug_feature_null": str(path.relative_to(ROOT)),
    }


def extract_lincs() -> dict:
    """LINCS between-drug global r from lincs_signatures."""
    path = EXP / "04_solutions" / "lincs" / "01_lincs_signatures" / "report" / "data" / "metrics.json"
    d = load(path)
    return {
        "morganBaselineGlobalR": round(d["morgan_fp"]["global_r_mean"], 3),
        "lincsGlobalR": round(d["lincs_sig"]["global_r_mean"], 3),
        "lincsGlobalRLift": round(d["lincs_sig"]["global_r_mean"] - d["morgan_fp"]["global_r_mean"], 3),
        "lincsPerDrugR": round(d["lincs_sig"]["per_drug_r_mean"], 4),
        "_src_lincs": str(path.relative_to(ROOT)),
    }


def extract_response_matching() -> dict:
    """K-curve cell-mean prior and K=50 blended from response_matching."""
    path = EXP / "04_solutions" / "response_matching" / "03_response_matching" / "report" / "data" / "metrics.json"
    d = load(path)
    cell_mean = d["0"]["cell_mean"]["mean"]
    k50_blended = d["50"]["blended"]["mean"]
    return {
        "cellMeanPriorPerDrugR": round(cell_mean, 3),
        "kFiftyBlendedPerDrugR": round(k50_blended, 3),
        "kFiftyLiftVsCellMean": round(k50_blended - cell_mean, 3),
        "_src_response_matching": str(path.relative_to(ROOT)),
    }


def extract_moa_stratified() -> dict:
    """ERK MAPK and EGFR MoA-stratified per-drug r (multi-seed)."""
    path = EXP / "04_solutions" / "moa_stratified" / "03_within_moa_training" / "validation" / "multiseed_results.json"
    d = load(path)
    erk = d["ERK MAPK signaling"]
    egfr = d["EGFR signaling"]
    return {
        "erkMapkAllDrugPerDrugR": round(erk["phase32_r"], 3),
        "erkMapkMoaStratifiedPerDrugR": round(erk["within_moa_mean"], 3),
        "erkMapkMoaStratifiedStd": round(erk["within_moa_std"], 3),
        "erkMapkMoaStratifiedCi90Low": round(erk["ci90"][0], 3),
        "erkMapkMoaStratifiedCi90High": round(erk["ci90"][1], 3),
        "egfrAllDrugPerDrugR": round(egfr["phase32_r"], 3),
        "egfrMoaStratifiedPerDrugR": round(egfr["within_moa_mean"], 3),
        "_src_moa_stratified": str(path.relative_to(ROOT)),
    }


def extract_beataml() -> dict:
    """BeatAML drug-feature delta and K-shot from external validation."""
    path = EXP / "06_external_validation" / "03_beataml_validation" / "report" / "data" / "metrics.json"
    d = load(path)
    dn = d["drug_null"]
    ks = d["kshot"]
    return {
        "beatAmlNoDrugPerDrugR": round(dn["no_drug"], 3),
        "beatAmlMorganDelta": round(dn["delta"], 3),
        "beatAmlKZeroPerDrugR": round(ks["K0"]["mean"], 3),
        "beatAmlKFiftyPerDrugR": round(ks["K50"]["mean"], 3),
        "beatAmlKFiftyLift": round(ks["lift_K50"], 3),
        "_src_beataml": str(path.relative_to(ROOT)),
    }


def extract_paso() -> dict:
    """PASO artifact decomposition."""
    path = EXP / "02_reproductions/01_paso/02_decomposition" / "report" / "data" / "metrics.json"
    d = load(path)
    return {
        "pasoFairMean": round(d["fair_mean"], 3),
        "pasoStyleMean": round(d["paso_style_mean"], 3),
        "pasoInflation": round(d["inflation"], 3),
        "pasoBestFoldTestR": round(d["best_fold_test_r"], 3),
        "_src_paso": str(path.relative_to(ROOT)),
    }


# ── main ───────────────────────────────────────────────────────────────────

ADAPTERS = [
    ("drug_feature_null", extract_drug_feature_null),
    ("lincs", extract_lincs),
    ("response_matching", extract_response_matching),
    ("moa_stratified", extract_moa_stratified),
    ("beataml", extract_beataml),
    ("paso", extract_paso),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Print only, do not write")
    args = parser.parse_args()

    metrics: dict = {}
    errors: list[str] = []

    for name, fn in ADAPTERS:
        try:
            result = fn()
            metrics.update(result)
            print(f"[OK] {name}: {len([k for k in result if not k.startswith('_')])} values extracted")
        except Exception as e:
            errors.append(f"[FAIL] {name}: {e}")
            print(errors[-1])

    print(f"\nTotal metrics: {len([k for k in metrics if not k.startswith('_')])}")

    if args.check:
        print("\nValues (check mode — not writing):")
        for k, v in sorted(metrics.items()):
            if not k.startswith("_"):
                print(f"  {k}: {v}")
        if errors:
            sys.exit(1)
        return

    if errors:
        print(f"\n{len(errors)} extractor(s) failed. Aborting write.")
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
