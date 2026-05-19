"""Step 3: Aggregate fold results into final metric_analysis.json.

Reads per-fold intermediates from results/fold_metrics/.
Writes results/per_drug_metrics.parquet and results/metric_analysis.json.

Runtime: < 30 sec CPU.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parents[4]
EXP_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

FOLD_DIR = EXP_DIR / "results" / "fold_metrics"
RESULTS_DIR = EXP_DIR / "results"
N_FOLDS = 5
METRIC_NAMES = ["r_p", "r_s", "tau", "ndcg5", "r2"]
METRIC_2A_NAMES = [f"{m}_2a" for m in METRIC_NAMES]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def aggregate_ci_widths(all_widths: dict[str, list[float]]) -> dict:
    result: dict[str, dict] = {}
    for m, widths in all_widths.items():
        if widths:
            arr = np.array(widths)
            result[m] = {
                "median": round(float(np.median(arr)), 6),
                "iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 6),
                "n_drugs_folds": len(arr),
            }
        else:
            result[m] = {"median": None, "iqr": None, "n_drugs_folds": 0}
    return result


def aggregate_predictor_2a(per_drug_df: pd.DataFrame) -> dict:
    """Median per-drug metric values for the constant drug-mean predictor."""
    result: dict[str, dict] = {}
    for m in METRIC_NAMES:
        col = f"{m}_2a"
        if col not in per_drug_df.columns:
            result[m] = {"median": None, "n": 0}
            continue
        vals = per_drug_df[col].dropna()
        result[m] = {
            "median": round(float(vals.median()), 6) if len(vals) > 0 else None,
            "n": len(vals),
        }
    return result


def compute_inter_metric_correlation(per_drug_df: pd.DataFrame) -> dict[str, float | None]:
    corr: dict[str, float | None] = {}
    for i, m1 in enumerate(METRIC_NAMES):
        for m2 in METRIC_NAMES[i + 1:]:
            col1 = per_drug_df[m1].dropna()
            col2 = per_drug_df[m2].dropna()
            shared = col1.index.intersection(col2.index)
            if len(shared) < 10:
                corr[f"{m1}_vs_{m2}"] = None
                continue
            r, _ = spearmanr(col1.loc[shared], col2.loc[shared])
            corr[f"{m1}_vs_{m2}"] = round(float(r), 4)
    return corr


def main() -> None:
    # Load per-fold per-drug metrics
    dfs = []
    missing = []
    for i in range(N_FOLDS):
        path = FOLD_DIR / f"fold{i}_per_drug.parquet"
        if not path.exists():
            missing.append(i)
            continue
        dfs.append(pd.read_parquet(path))
    if missing:
        raise FileNotFoundError(
            f"Missing fold_metrics for folds {missing}. "
            "Run compute_fold_metrics.py for each fold first."
        )

    per_drug_df = pd.concat(dfs, ignore_index=True)
    per_drug_df.to_parquet(RESULTS_DIR / "per_drug_metrics.parquet", index=False)
    log.info("Saved per_drug_metrics.parquet (%d rows)", len(per_drug_df))

    # Aggregate bootstrap CI widths across folds
    all_widths: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for i in range(N_FOLDS):
        path = FOLD_DIR / f"fold{i}_bootstrap.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with path.open() as f:
            fold_widths = json.load(f)
        for m in METRIC_NAMES:
            all_widths[m].extend(fold_widths.get(m, []))

    ci_summary = aggregate_ci_widths(all_widths)
    rp_med = ci_summary["r_p"]["median"]
    rs_med = ci_summary["r_s"]["median"]
    ratio = round(rp_med / rs_med, 4) if (rp_med and rs_med and rs_med > 0) else None

    # Inter-metric Spearman correlation
    inter_metric = compute_inter_metric_correlation(per_drug_df)
    log.info("r_p vs r_s (Spearman): %s", inter_metric.get("r_p_vs_r_s"))
    log.info("Pearson/Spearman CI ratio: %s", ratio)

    # Predictor 2A sanity check (constant drug-mean predictor)
    predictor_2a = aggregate_predictor_2a(per_drug_df)
    log.info("Predictor 2A r_p median: %s  (expected 0.0)", predictor_2a["r_p"]["median"])
    log.info("Predictor 2A ndcg5 median: %s", predictor_2a["ndcg5"]["median"])

    output = {
        "ci_width": ci_summary,
        "pearson_spearman_ci_ratio": ratio,
        "inter_metric_spearman": inter_metric,
        "predictor_2a": predictor_2a,
        "n_drugs_per_fold": {
            str(i): int((per_drug_df["fold"] == i).sum()) for i in range(N_FOLDS)
        },
    }
    out_path = RESULTS_DIR / "metric_analysis.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved → %s", out_path)

    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "metric_analysis.json"
    with report_path.open("w") as f:
        json.dump(output, f, indent=2)
    log.info("Report data → %s", report_path)

    # Print recommendation
    if ratio is None:
        log.info("RECOMMENDATION: insufficient data.")
    elif ratio > 1.1:
        log.info("RECOMMENDATION: CI ratio=%.3f > 1.1 → per-drug Spearman r.", ratio)
    else:
        log.info("RECOMMENDATION: CI ratio=%.3f ≤ 1.1 → per-drug Pearson r.", ratio)


if __name__ == "__main__":
    main()
