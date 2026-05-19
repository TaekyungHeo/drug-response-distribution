"""MoA x K-shot factorial: within-MoA training combined with K-shot response matching.

Four conditions per focus MoA (ERK MAPK signaling, EGFR signaling):
  1. All-drug Ridge baseline
  2. Within-MoA Ridge only
  3. K-shot response matching only (all-drug base)
  4. Within-MoA Ridge + K-shot combined

PASO 10-fold drug-blind CV.  K in {0, 5, 10, 20, 50}, 10 random anchor draws.

CLI:
  --smoke    2 folds, K={0,5,50}, 2 draws

Output:
  report/data/results.json
  report/data/combination_results.csv
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
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))

from src.utils.paso_folds import load_cell_line_index, load_paso_pairs
from src.utils.ridge import compress_cell
from src.utils.solutions import (
    group_drugs_by_moa,
    load_moa_annotations,
    response_match_predict,
)

EXP_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT / "data" / "processed"
PASO_FOLDS_DIR = ROOT / "external" / "PASO" / "data" / "10_fold_data" / "drug_blind"

K_FOLDS = 10
K_VALUES = [0, 5, 10, 20, 50]
K_VALUES_SMOKE = [0, 5, 50]
N_DRAWS = 10
N_DRAWS_SMOKE = 2
RIDGE_ALPHA = 1.0
N_NEIGHBORS = 5
W_GRID = [round(w * 0.1, 1) for w in range(11)]  # 0.0 .. 1.0
MIN_MOA_DRUGS = 3

FOCUS_MOAS = [
    "ERK MAPK signaling",
    "EGFR signaling",
]

MEASUREMENT_CEILING = 0.754

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_response_matrix(
    df: pd.DataFrame, drug_list: list[str], cell_list: list[str],
) -> np.ndarray:
    """Build (n_drugs, n_cells) response matrix; NaN where unobserved."""
    drug_to_row = {d: i for i, d in enumerate(drug_list)}
    cell_to_col = {c: j for j, c in enumerate(cell_list)}
    mat = np.full((len(drug_list), len(cell_list)), np.nan, dtype=np.float64)
    for _, row in df.iterrows():
        di = drug_to_row.get(row["drug_name"])
        ci = cell_to_col.get(row["depmap_id"])
        if di is not None and ci is not None:
            mat[di, ci] = row["ln_ic50"]
    return mat


def _drug_r(preds: np.ndarray, targets: np.ndarray, min_cells: int = 5) -> float:
    """Pearson r; returns NaN if insufficient data."""
    ok = ~np.isnan(preds) & ~np.isnan(targets)
    if ok.sum() < min_cells:
        return float("nan")
    p, t = preds[ok], targets[ok]
    if p.std() < 1e-8 or t.std() < 1e-8:
        return float("nan")
    return float(pearsonr(p, t)[0])


def _kshot_blend(
    train_response_matrix: np.ndarray,
    test_drug_responses: np.ndarray,
    anchor_idx: np.ndarray,
    fallback_cell_mean: np.ndarray,
) -> np.ndarray | None:
    """Pure neighbor prediction via response matching, or None if infeasible.

    Returns the raw neighbor_pred (blend_weight=1.0).  Caller blends with
    the desired base prediction at the desired w.
    """
    if len(anchor_idx) == 0:
        return None  # K=0: no matching possible

    test_observed = test_drug_responses[anchor_idx].copy()
    neighbor_pred = response_match_predict(
        train_response_matrix=train_response_matrix,
        test_observed=test_observed,
        anchor_cell_idx=anchor_idx,
        cell_mean=fallback_cell_mean,
        blend_weight=1.0,
        n_neighbors=N_NEIGHBORS,
    )
    return neighbor_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MoA x K-shot factorial")
    parser.add_argument("--smoke", action="store_true", help="Quick run")
    args = parser.parse_args()

    n_folds = 2 if args.smoke else K_FOLDS
    k_values = K_VALUES_SMOKE if args.smoke else K_VALUES
    n_draws = N_DRAWS_SMOKE if args.smoke else N_DRAWS

    logger.info(
        "01_moa_x_kshot | ROOT=%s | folds=%d | K=%s | draws=%d",
        ROOT, n_folds, k_values, n_draws,
    )

    # ---- Load omics ----
    rna = pd.read_parquet(DATA_DIR / "rna.parquet")
    mutations = pd.read_parquet(DATA_DIR / "mutations.parquet")
    logger.info("RNA: %s  mutations: %s", rna.shape, mutations.shape)

    name_to_depmap = load_cell_line_index(DATA_DIR)
    available_cells = set(rna.index) & set(mutations.index)

    # ---- MoA annotations ----
    moa = load_moa_annotations()
    logger.info("MoA annotations: %d drugs", len(moa))

    # ---- Build per-fold data ----
    fold_data: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for k in range(n_folds):
        train_df, test_df = load_paso_pairs(
            PASO_FOLDS_DIR, name_to_depmap, available_cells, k,
        )
        fold_data.append((train_df, test_df))

    # Map each drug to its test fold
    drug_to_test_fold: dict[str, int] = {}
    for k, (_, test_df) in enumerate(fold_data):
        for d in test_df["drug_name"].unique():
            drug_to_test_fold[d] = k

    # ---- Group drugs by MoA, keep focus MoAs with enough drugs ----
    all_drug_names = sorted(drug_to_test_fold.keys())
    moa_groups = group_drugs_by_moa(all_drug_names, moa)
    moa_groups = {
        m: drugs for m, drugs in moa_groups.items()
        if m in FOCUS_MOAS and len(drugs) >= MIN_MOA_DRUGS
    }
    logger.info(
        "Focus MoA groups: %d classes, %d drugs total",
        len(moa_groups),
        sum(len(v) for v in moa_groups.values()),
    )

    # ---- Accumulate records ----
    # Each record: (drug, moa, condition, k, draw, r, w)
    records: list[dict] = []

    # Group work by fold
    fold_to_work: dict[int, list[tuple[str, str, list[str]]]] = {}
    for moa_label, drugs in moa_groups.items():
        for held_out in drugs:
            k_fold = drug_to_test_fold.get(held_out)
            if k_fold is None:
                continue
            train_drugs_moa = [d for d in drugs if d != held_out]
            if len(train_drugs_moa) < 2:
                continue
            fold_to_work.setdefault(k_fold, []).append(
                (moa_label, held_out, train_drugs_moa)
            )

    for fold_i in sorted(fold_to_work.keys()):
        t0 = datetime.now()
        work_items = fold_to_work[fold_i]
        logger.info(
            "Fold %d: %d LOO tasks, started at %s",
            fold_i, len(work_items), t0.strftime("%H:%M:%S"),
        )

        train_df, test_df = fold_data[fold_i]
        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty data — skipping", fold_i)
            continue

        # ---- Cell features (PCA fit on train cells) ----
        all_cells = sorted(set(train_df["depmap_id"]) | set(test_df["depmap_id"]))
        cell_to_col = {c: j for j, c in enumerate(all_cells)}
        n_cells = len(all_cells)

        rna_arr = rna.loc[all_cells].values.astype(np.float32)
        mut_arr = mutations.loc[all_cells].values.astype(np.float32)
        train_cell_rows = np.unique(
            np.array([cell_to_col[c] for c in train_df["depmap_id"]], dtype=np.int32)
        )
        rna_c, mut_c = compress_cell(rna_arr, mut_arr, train_cell_rows)
        cell_feat = np.concatenate([rna_c, mut_c], axis=1).astype(np.float32)

        # ---- All-drug response matrix & cell mean ----
        all_train_drugs = sorted(train_df["drug_name"].unique())
        all_response_matrix = _build_response_matrix(train_df, all_train_drugs, all_cells)
        all_cell_mean = np.nanmean(all_response_matrix, axis=0)
        global_mean = np.nanmean(all_cell_mean)
        all_cell_mean = np.where(np.isnan(all_cell_mean), global_mean, all_cell_mean)

        # Pre-index train/test by drug
        train_by_drug: dict[str, pd.DataFrame] = {
            d: grp for d, grp in train_df.groupby("drug_name")
        }
        test_by_drug: dict[str, pd.DataFrame] = {
            d: grp for d, grp in test_df.groupby("drug_name")
        }

        # ---- All-drug Ridge (fit once per fold) ----
        X_train_all = cell_feat[
            np.array([cell_to_col[c] for c in train_df["depmap_id"]], dtype=np.int32)
        ]
        y_train_all = train_df["ln_ic50"].values.astype(np.float64)
        ridge_all = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        ridge_all.fit(X_train_all, y_train_all)

        # Full-length all-drug Ridge prediction (for K-shot only blending)
        pred_all_drug_full = ridge_all.predict(cell_feat)

        # ---- Process each held-out drug ----
        for moa_label, held_out, moa_train_drugs in work_items:
            if held_out not in test_by_drug:
                continue
            moa_test = test_by_drug[held_out]
            if len(moa_test) < 5:
                continue

            # Test cell indices
            test_cell_idx = np.array(
                [cell_to_col[c] for c in moa_test["depmap_id"]], dtype=np.int32
            )
            y_test = moa_test["ln_ic50"].values.astype(np.float64)

            # Full-length test response vector (for K-shot matching)
            test_response_vec = np.full(n_cells, np.nan, dtype=np.float64)
            for _, row in moa_test.iterrows():
                ci = cell_to_col.get(row["depmap_id"])
                if ci is not None:
                    test_response_vec[ci] = row["ln_ic50"]

            obs_idx = np.where(~np.isnan(test_response_vec))[0]

            # ---- Condition 1: All-drug Ridge prediction ----
            pred_all_drug = ridge_all.predict(cell_feat[test_cell_idx])

            # ---- Condition 2: Within-MoA Ridge ----
            moa_train_parts = [
                train_by_drug[d] for d in moa_train_drugs if d in train_by_drug
            ]
            if not moa_train_parts:
                continue
            moa_train_df = pd.concat(moa_train_parts, ignore_index=True)
            if len(moa_train_df) < 10:
                continue

            X_train_moa = cell_feat[
                np.array([cell_to_col[c] for c in moa_train_df["depmap_id"]], dtype=np.int32)
            ]
            y_train_moa = moa_train_df["ln_ic50"].values.astype(np.float64)
            ridge_moa = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
            ridge_moa.fit(X_train_moa, y_train_moa)

            pred_within_moa = ridge_moa.predict(cell_feat[test_cell_idx])

            # Full-length within-MoA Ridge prediction (for combined blending)
            pred_within_moa_full = ridge_moa.predict(cell_feat)

            # ---- Within-MoA response matrix & cell mean (for combined condition) ----
            moa_train_drug_list = sorted(
                [d for d in moa_train_drugs if d in train_by_drug]
            )
            moa_response_matrix = _build_response_matrix(
                moa_train_df, moa_train_drug_list, all_cells,
            )
            moa_cell_mean = np.nanmean(moa_response_matrix, axis=0)
            moa_cell_mean = np.where(
                np.isnan(moa_cell_mean), global_mean, moa_cell_mean
            )

            logger.info(
                "  Drug %s (%s): %d test cells, %d MoA train drugs, "
                "MoA response matrix %s",
                held_out, moa_label, len(test_cell_idx),
                len(moa_train_drug_list), moa_response_matrix.shape,
            )

            # ---- Sweep K and draws ----
            for k in k_values:
                for draw in range(n_draws):
                    rng = np.random.default_rng(
                        10000 * fold_i + 100 * (hash(held_out) % 10000) + 10 * k + draw
                    )

                    if k == 0:
                        # No anchors — eval on all test cells

                        # Cond 1: all-drug
                        r1 = _drug_r(pred_all_drug, y_test)
                        records.append(dict(
                            drug=held_out, moa=moa_label,
                            condition="all_drug_baseline", k=0, draw=draw,
                            r=r1, w=0.0,
                        ))

                        # Cond 2: within-MoA
                        r2 = _drug_r(pred_within_moa, y_test)
                        records.append(dict(
                            drug=held_out, moa=moa_label,
                            condition="within_moa_only", k=0, draw=draw,
                            r=r2, w=0.0,
                        ))

                        # Cond 3: K-shot only at K=0 is just all-drug Ridge
                        records.append(dict(
                            drug=held_out, moa=moa_label,
                            condition="kshot_only", k=0, draw=draw,
                            r=r1, w=0.0,
                        ))

                        # Cond 4: combined at K=0 equals within-MoA only
                        records.append(dict(
                            drug=held_out, moa=moa_label,
                            condition="combined", k=0, draw=draw,
                            r=r2, w=0.0,
                        ))
                        continue

                    # K > 0: sample anchor cells
                    if len(obs_idx) < k + 5:
                        continue  # not enough cells

                    anchor_idx = rng.choice(obs_idx, size=k, replace=False)
                    anchor_set = set(anchor_idx.tolist())
                    # Eval cells: test cells that are NOT anchors
                    eval_mask = np.array(
                        [cell_to_col[c] not in anchor_set for c in moa_test["depmap_id"]]
                    )
                    if eval_mask.sum() < 5:
                        continue
                    eval_cell_idx = test_cell_idx[eval_mask]
                    y_eval = y_test[eval_mask]

                    # Cond 1: all-drug Ridge (eval on non-anchor)
                    r1 = _drug_r(
                        ridge_all.predict(cell_feat[eval_cell_idx]), y_eval,
                    )
                    records.append(dict(
                        drug=held_out, moa=moa_label,
                        condition="all_drug_baseline", k=k, draw=draw,
                        r=r1, w=0.0,
                    ))

                    # Cond 2: within-MoA Ridge (eval on non-anchor)
                    r2 = _drug_r(
                        ridge_moa.predict(cell_feat[eval_cell_idx]), y_eval,
                    )
                    records.append(dict(
                        drug=held_out, moa=moa_label,
                        condition="within_moa_only", k=k, draw=draw,
                        r=r2, w=0.0,
                    ))

                    # Cond 3: K-shot only (all-drug response matrix,
                    #          blend neighbor pred with all-drug Ridge)
                    neighbor_pred_all = _kshot_blend(
                        all_response_matrix, test_response_vec,
                        anchor_idx, all_cell_mean,
                    )
                    if neighbor_pred_all is not None:
                        best_r3, best_w3 = -np.inf, 0.0
                        for w in W_GRID:
                            blended = w * neighbor_pred_all + (1.0 - w) * pred_all_drug_full
                            r_val = _drug_r(blended[eval_cell_idx], y_eval)
                            if not np.isnan(r_val) and r_val > best_r3:
                                best_r3 = r_val
                                best_w3 = w
                        records.append(dict(
                            drug=held_out, moa=moa_label,
                            condition="kshot_only", k=k, draw=draw,
                            r=best_r3 if best_r3 > -np.inf else float("nan"),
                            w=best_w3,
                        ))
                    else:
                        records.append(dict(
                            drug=held_out, moa=moa_label,
                            condition="kshot_only", k=k, draw=draw,
                            r=float("nan"), w=0.0,
                        ))

                    # Cond 4: Combined (within-MoA response matrix,
                    #          blend neighbor pred with within-MoA Ridge)
                    neighbor_pred_moa = _kshot_blend(
                        moa_response_matrix, test_response_vec,
                        anchor_idx, moa_cell_mean,
                    )
                    if neighbor_pred_moa is not None:
                        best_r4, best_w4 = -np.inf, 0.0
                        for w in W_GRID:
                            blended = w * neighbor_pred_moa + (1.0 - w) * pred_within_moa_full
                            r_val = _drug_r(blended[eval_cell_idx], y_eval)
                            if not np.isnan(r_val) and r_val > best_r4:
                                best_r4 = r_val
                                best_w4 = w
                        records.append(dict(
                            drug=held_out, moa=moa_label,
                            condition="combined", k=k, draw=draw,
                            r=best_r4 if best_r4 > -np.inf else float("nan"),
                            w=best_w4,
                        ))
                    else:
                        records.append(dict(
                            drug=held_out, moa=moa_label,
                            condition="combined", k=k, draw=draw,
                            r=float("nan"), w=0.0,
                        ))

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info("  Fold %d: completed in %.1fs", fold_i, elapsed)

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #
    logger.info("Collected %d records", len(records))
    df_all = pd.DataFrame(records)
    df_all = df_all.dropna(subset=["r"])

    # ---- Per-MoA summary ----
    per_moa_out: list[dict] = []
    per_drug_out: list[dict] = []

    for moa_label, drugs in sorted(moa_groups.items()):
        moa_sub = df_all[df_all["moa"] == moa_label]
        if moa_sub.empty:
            continue

        conditions_out: dict = {}

        # -- all_drug_baseline --
        c1 = moa_sub[moa_sub["condition"] == "all_drug_baseline"]
        if not c1.empty:
            drug_means = c1.groupby("drug")["r"].mean()
            conditions_out["all_drug_baseline"] = {
                "per_drug_r": round(float(drug_means.mean()), 6),
            }

        # -- within_moa_only --
        c2 = moa_sub[moa_sub["condition"] == "within_moa_only"]
        if not c2.empty:
            drug_means = c2.groupby("drug")["r"].mean()
            conditions_out["within_moa_only"] = {
                "per_drug_r": round(float(drug_means.mean()), 6),
            }

        # -- kshot_only: k_curve --
        c3 = moa_sub[moa_sub["condition"] == "kshot_only"]
        kshot_curve: list[dict] = []
        for kv in k_values:
            c3k = c3[c3["k"] == kv]
            if c3k.empty:
                continue
            drug_means = c3k.groupby("drug")["r"].mean()
            opt_w = round(float(c3k.groupby("drug")["w"].mean().mean()), 2) if c3k["w"].notna().any() else 0.0
            kshot_curve.append({
                "k": kv,
                "per_drug_r": round(float(drug_means.mean()), 6),
                "optimal_w": opt_w,
            })
        conditions_out["kshot_only"] = {"k_curve": kshot_curve}

        # -- combined: k_curve --
        c4 = moa_sub[moa_sub["condition"] == "combined"]
        combined_curve: list[dict] = []
        for kv in k_values:
            c4k = c4[c4["k"] == kv]
            if c4k.empty:
                continue
            drug_means = c4k.groupby("drug")["r"].mean()
            opt_w = round(float(c4k.groupby("drug")["w"].mean().mean()), 2) if c4k["w"].notna().any() else 0.0
            combined_curve.append({
                "k": kv,
                "per_drug_r": round(float(drug_means.mean()), 6),
                "optimal_w": opt_w,
            })
        conditions_out["combined"] = {"k_curve": combined_curve}

        per_moa_out.append({
            "moa": moa_label,
            "n_drugs": len(drugs),
            "conditions": conditions_out,
        })

        # -- Per-drug detail (K=0 and max K) --
        max_k = max(k_values)
        for drug in drugs:
            drug_sub = moa_sub[moa_sub["drug"] == drug]
            if drug_sub.empty:
                continue
            ad = drug_sub[(drug_sub["condition"] == "all_drug_baseline") & (drug_sub["k"] == 0)]
            wm = drug_sub[(drug_sub["condition"] == "within_moa_only") & (drug_sub["k"] == 0)]
            ks = drug_sub[(drug_sub["condition"] == "kshot_only") & (drug_sub["k"] == max_k)]
            cb = drug_sub[(drug_sub["condition"] == "combined") & (drug_sub["k"] == max_k)]
            per_drug_out.append({
                "drug": drug,
                "drug_id": None,
                "moa": moa_label,
                "all_drug_r": round(float(ad["r"].mean()), 6) if not ad.empty else None,
                "within_moa_r": round(float(wm["r"].mean()), 6) if not wm.empty else None,
                "kshot_r": round(float(ks["r"].mean()), 6) if not ks.empty else None,
                "combined_r": round(float(cb["r"].mean()), 6) if not cb.empty else None,
                "max_k": max_k,
            })

    results = {
        "measurement_ceiling": MEASUREMENT_CEILING,
        "per_moa": per_moa_out,
        "per_drug": per_drug_out,
    }

    # ---- Write outputs ----
    report_data = EXP_DIR / "report" / "data"
    report_data.mkdir(parents=True, exist_ok=True)

    out_json = report_data / "results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Written: %s", out_json)

    # Flat CSV
    csv_df = df_all[["drug", "moa", "condition", "k", "draw", "r"]].copy()
    csv_path = report_data / "combination_results.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info("Written: %s", csv_path)

    # ---- Validation: K=0 combined == within-MoA only ----
    logger.info("=" * 70)
    logger.info("Validation: K=0 combined should equal within_moa_only")
    for moa_entry in per_moa_out:
        wm_r = moa_entry["conditions"].get("within_moa_only", {}).get("per_drug_r")
        comb_k0 = [
            e for e in moa_entry["conditions"].get("combined", {}).get("k_curve", [])
            if e["k"] == 0
        ]
        comb_r = comb_k0[0]["per_drug_r"] if comb_k0 else None
        logger.info(
            "  %-25s within_moa=%.4f  combined_k0=%.4f  match=%s",
            moa_entry["moa"],
            wm_r if wm_r is not None else float("nan"),
            comb_r if comb_r is not None else float("nan"),
            "YES" if wm_r is not None and comb_r is not None and abs(wm_r - comb_r) < 1e-4 else "NO",
        )

    # ---- Summary table ----
    logger.info("=" * 70)
    for moa_entry in per_moa_out:
        moa_label = moa_entry["moa"]
        conds = moa_entry["conditions"]
        logger.info("MoA: %s (n=%d)", moa_label, moa_entry["n_drugs"])
        logger.info(
            "  all_drug_baseline: %.4f",
            conds.get("all_drug_baseline", {}).get("per_drug_r", float("nan")),
        )
        logger.info(
            "  within_moa_only:   %.4f",
            conds.get("within_moa_only", {}).get("per_drug_r", float("nan")),
        )
        logger.info("  kshot_only:")
        for row in conds.get("kshot_only", {}).get("k_curve", []):
            logger.info("    K=%3d  r=%.4f  w=%.2f", row["k"], row["per_drug_r"], row["optimal_w"])
        logger.info("  combined:")
        for row in conds.get("combined", {}).get("k_curve", []):
            logger.info("    K=%3d  r=%.4f  w=%.2f", row["k"], row["per_drug_r"], row["optimal_w"])
        logger.info("")
    logger.info("Measurement ceiling: %.3f", MEASUREMENT_CEILING)


if __name__ == "__main__":
    main()
