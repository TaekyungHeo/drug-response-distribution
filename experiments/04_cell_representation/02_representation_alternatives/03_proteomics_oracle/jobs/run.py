"""05_proteomics_oracle: does RPPA proteomics break the per-drug r ceiling?

Tests whether protein-level measurements (RPPA, 214 proteins) carry information
beyond RNA+mut, especially for Apoptosis regulation (claimed as 'genuine biological limit').

Conditions (all Ridge(α=1.0), PASO 10-fold drug-blind CV on RPPA-covered cells):
  A_rna_mut    : RNA PCA(550) + mut PCA(200)  [baseline restricted to RPPA cells]
  B_rppa       : RPPA alone (214 proteins, no PCA — fewer features than RNA)
  C_rna_mut_rppa: RNA + mut + RPPA concatenated

Usage:
  python run.py                         # all conditions → results.json
  python run.py --condition A_rna_mut  # single → results_A_rna_mut.json
  python run.py --smoke                 # 1 fold only

Output: report/data/results.json  (or results_<condition>.json)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.evaluation.per_drug import per_drug_r  # noqa: E402
from src.utils.paso_folds import load_paso_pairs  # noqa: E402
from src.utils.ridge import compress_cell, safe_fit_scaler  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"
EXP_DIR = Path(__file__).parents[1]

K_FOLDS = 10
MIN_CELLS = 5
RNA_DIM, MUT_DIM = 550, 200

FOCUS_MOA = [
    "Apoptosis regulation",
    "ERK MAPK signaling",
    "EGFR signaling",
    "PI3K/MTOR signaling",
    "Mitosis",
]

ALL_CONDITIONS = ["A_rna_mut", "B_rppa", "C_rna_mut_rppa"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 fold only")
    parser.add_argument("--condition", type=str, choices=ALL_CONDITIONS, default=None,
                        help="Run a single condition (writes results_<condition>.json)")
    args = parser.parse_args()
    k_folds = 1 if args.smoke else K_FOLDS
    active_conditions = [args.condition] if args.condition else ALL_CONDITIONS

    report_dir = EXP_DIR / "report" / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    fh = logging.FileHandler(logs_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("05_proteomics_oracle: RPPA vs RNA per-drug r, focus on Apoptosis%s%s",
                f" [cond={args.condition}]" if args.condition else "",
                " [SMOKE]" if args.smoke else "")

    if args.condition:
        results_path = report_dir / f"results_{args.condition}.json"
    else:
        results_path = report_dir / "results.json"

    # Load RPPA (CELLLINE_TISSUE index → strip tissue, uppercase)
    rppa_raw = pd.read_csv(ROOT / "data" / "raw" / "CCLE_RPPA_20181003.csv", index_col=0)
    rppa_raw.index = rppa_raw.index.str.split("_").str[0].str.upper()
    rppa_raw = rppa_raw[~rppa_raw.index.duplicated(keep="first")]  # drop duplicate cell entries
    rppa_raw = rppa_raw.fillna(rppa_raw.mean())
    logger.info("RPPA: %d cells × %d proteins", rppa_raw.shape[0], rppa_raw.shape[1])

    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")

    # Build depmap_to_stripped for RPPA lookup
    cl_idx = pd.read_parquet(DATA_DIR / "cell_line_index.parquet")
    name_to_depmap: dict[str, str] = {}
    depmap_to_stripped: dict[str, str] = {}
    for depmap_id, row in cl_idx.iterrows():
        dep = str(depmap_id)
        stripped = str(row["stripped_name"]).upper()
        name_to_depmap[stripped] = dep
        depmap_to_stripped[dep] = stripped

    # available_cells: rna ∩ mutations ∩ rppa coverage
    rppa_cells = {dep for dep, stripped in depmap_to_stripped.items()
                  if stripped in rppa_raw.index}
    available_cells = set(rna.index) & set(mutations.index) & rppa_cells
    logger.info("RPPA-covered cells: %d  after rna∩mut filter: %d",
                len(rppa_cells), len(available_cells))

    # MoA lookup
    moa_raw = pd.read_csv(
        ROOT / "external" / "PASO" / "Figs" / "Fig7" / "GDSC2_Drug_Pathway_Target.csv"
    )
    drug_moa = dict(zip(
        moa_raw["Drug name"].astype(str).str.strip(),
        moa_raw["Target Pathway"].astype(str).str.strip(),
    ))

    fold_results: dict[str, list[dict]] = {c: [] for c in active_conditions}
    moa_fold_results: dict[str, dict[str, list[float]]] = {
        c: defaultdict(list) for c in active_conditions
    }

    for fold_i in range(k_folds):
        logger.info("=== Fold %d/%d ===", fold_i + 1, k_folds)
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, fold_i
        )
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("fold %d: empty split — skip", fold_i)
            continue

        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        train_cells = sorted(train_df["depmap_id"].unique())
        cell_to_row = {c: i for i, c in enumerate(all_cells)}

        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        n_rppa = rppa_raw.shape[1]
        rppa_arr = np.zeros((len(all_cells), n_rppa), dtype=np.float32)
        for i, dep in enumerate(all_cells):
            stripped = depmap_to_stripped.get(dep, "")
            if stripped in rppa_raw.index:
                rppa_arr[i] = rppa_raw.loc[stripped].values.astype(np.float32)
        zero_rows = int((rppa_arr.sum(axis=1) == 0).sum())
        if zero_rows > 0:
            logger.warning("  fold %d: %d cells with zero RPPA row", fold_i, zero_rows)

        train_rows = np.array([cell_to_row[c] for c in train_cells], dtype=np.int32)
        tr_idx = np.array([cell_to_row[c] for c in train_df["depmap_id"]], dtype=np.int32)
        te_idx = np.array([cell_to_row[c] for c in test_df["depmap_id"]], dtype=np.int32)
        y_train = train_df["ln_ic50"].values.astype(np.float32)
        y_test = test_df["ln_ic50"].values.astype(np.float32)
        d_te = test_df["drug_name"].values

        logger.info("  train=%d pairs  test=%d pairs  train_cells=%d",
                    len(train_df), len(test_df), len(train_cells))

        needs_rna_mut = any(c in active_conditions for c in ["A_rna_mut", "C_rna_mut_rppa"])
        rna_mut = None
        if needs_rna_mut:
            rna_pca, mut_pca = compress_cell(rna_arr, mut_arr, train_rows,
                                             rna_dim=RNA_DIM, mut_dim=MUT_DIM)
            rna_mut = np.c_[rna_pca, mut_pca]

        for cname in active_conditions:
            if cname == "A_rna_mut":
                assert rna_mut is not None
                Xtr = rna_mut[tr_idx]
                Xte = rna_mut[te_idx]
            elif cname == "B_rppa":
                Xtr = rppa_arr[tr_idx]
                Xte = rppa_arr[te_idx]
            elif cname == "C_rna_mut_rppa":
                assert rna_mut is not None
                Xtr = np.c_[rna_mut[tr_idx], rppa_arr[tr_idx]]
                Xte = np.c_[rna_mut[te_idx], rppa_arr[te_idx]]
            else:
                continue

            sc = safe_fit_scaler(Xtr)
            ridge = Ridge(alpha=1.0)
            ridge.fit(sc.transform(Xtr), y_train)
            preds = ridge.predict(sc.transform(Xte)).astype(np.float32)

            rs = per_drug_r(preds, y_test, d_te, min_cells=MIN_CELLS)
            overall = float(np.mean(list(rs.values()))) if rs else float("nan")
            fold_results[cname].append({"per_drug_r": overall})

            for moa in FOCUS_MOA:
                moa_drugs = {d for d in np.unique(d_te) if drug_moa.get(d) == moa}
                rs_moa = {d: r for d, r in rs.items() if d in moa_drugs}
                if rs_moa:
                    moa_fold_results[cname][moa].append(float(np.mean(list(rs_moa.values()))))

        logger.info("  fold %d: " + " | ".join(
            f"{c[:12]}: {fold_results[c][-1]['per_drug_r']:.4f}" for c in active_conditions
        ), fold_i)

        results_path.write_text(json.dumps({
            "condition": args.condition,
            "fold_results": fold_results,
            "moa_fold_results": {c: dict(v) for c, v in moa_fold_results.items()},
        }, indent=2))

    logger.info("=" * 60)
    base_vals = fold_results.get("A_rna_mut", [])
    base_mean = float(np.mean([f["per_drug_r"] for f in base_vals])) if base_vals else float("nan")

    summary: dict[str, dict] = {}
    for cname in active_conditions:
        vals = [f["per_drug_r"] for f in fold_results[cname]]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals))
        delta = m - base_mean
        summary[cname] = {
            "per_drug_r_mean": round(m, 4),
            "per_drug_r_std": round(s, 4),
            "delta_vs_A": round(delta, 4) if not np.isnan(base_mean) else None,
        }
        logger.info("  %-20s  per-drug r=%.4f ± %.4f", cname, m, s)

    moa_summary: dict[str, dict[str, dict]] = {c: {} for c in active_conditions}
    base_moa = {
        moa: float(np.mean(moa_fold_results["A_rna_mut"][moa]))
        if "A_rna_mut" in moa_fold_results and moa_fold_results["A_rna_mut"][moa]
        else float("nan")
        for moa in FOCUS_MOA
    }
    for moa in FOCUS_MOA:
        for cname in active_conditions:
            vals = moa_fold_results[cname][moa]
            m = float(np.mean(vals)) if vals else float("nan")
            delta = m - base_moa[moa] if not np.isnan(base_moa[moa]) else float("nan")
            moa_summary[cname][moa] = {
                "mean": round(m, 4) if not np.isnan(m) else None,
                "delta_vs_A": round(delta, 4) if not np.isnan(delta) else None,
            }

    out: dict = {
        "condition": args.condition,
        "summary": summary,
        "moa_summary": moa_summary,
        "fold_results": fold_results,
        "moa_fold_results": {c: dict(v) for c, v in moa_fold_results.items()},
    }

    if len(active_conditions) > 1 and "A_rna_mut" in moa_summary and "B_rppa" in moa_summary:
        apop_a = moa_summary["A_rna_mut"].get("Apoptosis regulation", {}).get("mean") or float("nan")
        apop_b = moa_summary["B_rppa"].get("Apoptosis regulation", {}).get("mean") or float("nan")
        apop_c = moa_summary.get("C_rna_mut_rppa", {}).get("Apoptosis regulation", {}).get("mean") or float("nan")
        max_rppa = max(apop_b, apop_c) if not (np.isnan(apop_b) or np.isnan(apop_c)) else float("nan")
        if not np.isnan(max_rppa) and not np.isnan(apop_a):
            if max_rppa - apop_a > 0.05:
                verdict = "RPPA breaks Apoptosis ceiling — revise 'genuine biological limit' to 'RNA-representation limit'"
            elif abs(max_rppa - apop_a) <= 0.02:
                verdict = "RPPA comparable to RNA — 'genuine biological limit' claim strengthened"
            else:
                verdict = f"Marginal RPPA effect (Δ={max_rppa - apop_a:+.3f}) — interpret with caution"
        else:
            verdict = "insufficient data"
        logger.info("VERDICT: %s", verdict)
        out["apoptosis_verdict"] = verdict

    results_path.write_text(json.dumps(out, indent=2))
    logger.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
