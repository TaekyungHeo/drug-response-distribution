"""Convert results.json → report/data/metrics.json for template rendering."""

import json
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).parent
REPORT_DATA = EXP_DIR / "report" / "data"


def main() -> None:
    runs = sorted((EXP_DIR / "results").glob("run_*"))
    if not runs:
        raise FileNotFoundError("No run directories found")
    run_dir = runs[-1]

    with open(run_dir / "results.json") as f:
        raw = json.load(f)

    folds = raw["folds"]
    metrics = {
        "n_folds": len(folds),
        "paso_reported_r": 0.745,  # Wu et al. 2025 headline figure
        "paso_style_mean": round(raw["paso_style_mean"], 4),
        "paso_style_std": round(raw["paso_style_std"], 4),
        "fair_mean": round(raw["our_style_mean"], 4),
        "fair_std": round(raw["our_style_std"], 4),
        "inflation": round(raw["paso_style_mean"] - raw["our_style_mean"], 4),
        "best_fold_test_r": round(max(f["best_test_r"] for f in folds), 4),
        "best_fold_val_r": round(max(f["best_val_r"] for f in folds), 4),
        "fold_details": [
            {
                "best_test_r": round(f["best_test_r"], 4),
                "best_test_epoch": f["best_test_epoch"],
                "best_val_test_r": round(f["best_val_test_r"], 4),
                "fair_per_drug_r": round(f["fair_per_drug_r"], 4) if f.get("fair_per_drug_r") is not None else None,
                "paso_style_per_drug_r": round(f["paso_style_per_drug_r"], 4) if f.get("paso_style_per_drug_r") is not None else None,
            }
            for f in folds
        ],
    }

    if "paso_style_per_drug_mean" in raw:
        metrics["paso_style_per_drug_mean"] = round(raw["paso_style_per_drug_mean"], 4)
        metrics["paso_style_per_drug_std"] = round(raw["paso_style_per_drug_std"], 4)
    if "fair_per_drug_mean" in raw:
        metrics["fair_per_drug_mean"] = round(raw["fair_per_drug_mean"], 4)
        metrics["fair_per_drug_std"] = round(raw["fair_per_drug_std"], 4)

    REPORT_DATA.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DATA / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {REPORT_DATA / 'metrics.json'}")


if __name__ == "__main__":
    main()
