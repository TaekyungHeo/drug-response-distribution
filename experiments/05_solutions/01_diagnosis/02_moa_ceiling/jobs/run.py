"""02_moa_ceiling: Within-MoA biological ceiling for per-drug r.

Compute pairwise Pearson r between drug response profiles for drugs sharing
the same MoA class. Compare with a random-pair baseline (drug pairs sampled
regardless of MoA). High within-MoA r >> random baseline means MoA carries
real signal for within-MoA training.

Usage:
    uv run python3 experiments/05_solutions/01_diagnosis/02_moa_ceiling/jobs/run.py
    uv run python3 experiments/05_solutions/01_diagnosis/02_moa_ceiling/jobs/run.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).parents[5]
sys.path.insert(0, str(ROOT))
from src.utils.solutions import group_drugs_by_moa, load_moa_annotations

EXP_DIR = Path(__file__).parents[1]
OUT_DIR = EXP_DIR / "report" / "data"

MIN_SHARED_CELLS = 20
MIN_DRUGS_PER_GROUP = 3
RANDOM_SEED = 42


def build_response_matrix(
    dr: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pivot drug_response long table into (n_drugs, n_cells) matrix.

    Returns:
        response_matrix: (n_drugs, n_cells), NaN where unobserved.
        drug_names: list of drug names (row labels).
        cell_ids: list of depmap_ids (column labels).
    """
    pivot = dr.pivot_table(
        index="drug_name", columns="depmap_id", values="ln_ic50", aggfunc="mean"
    )
    drug_names = list(pivot.index)
    cell_ids = list(pivot.columns)
    matrix = pivot.values.astype(np.float64)
    return matrix, drug_names, cell_ids


def pairwise_r_for_group(
    response_matrix: np.ndarray,
    name_to_idx: dict[str, int],
    members: list[str],
    min_shared_cells: int = MIN_SHARED_CELLS,
) -> list[float]:
    """Compute all pairwise Pearson r between members in the response matrix."""
    idxs = [name_to_idx[d] for d in members if d in name_to_idx]
    pair_rs: list[float] = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            prof_a = response_matrix[idxs[a]]
            prof_b = response_matrix[idxs[b]]
            ok = ~np.isnan(prof_a) & ~np.isnan(prof_b)
            if ok.sum() < min_shared_cells:
                continue
            pa, pb = prof_a[ok], prof_b[ok]
            if pa.std() < 1e-8 or pb.std() < 1e-8:
                continue
            pair_rs.append(float(pearsonr(pa, pb)[0]))  # type: ignore[arg-type]
    return pair_rs


def compute_random_baseline(
    response_matrix: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    min_shared_cells: int = MIN_SHARED_CELLS,
) -> dict:
    """Sample random drug pairs and compute pairwise Pearson r."""
    n_drugs = response_matrix.shape[0]
    pair_rs: list[float] = []
    attempts = 0
    max_attempts = n_pairs * 10
    while len(pair_rs) < n_pairs and attempts < max_attempts:
        a, b = rng.choice(n_drugs, size=2, replace=False)
        attempts += 1
        prof_a = response_matrix[a]
        prof_b = response_matrix[b]
        ok = ~np.isnan(prof_a) & ~np.isnan(prof_b)
        if ok.sum() < min_shared_cells:
            continue
        pa, pb = prof_a[ok], prof_b[ok]
        if pa.std() < 1e-8 or pb.std() < 1e-8:
            continue
        pair_rs.append(float(pearsonr(pa, pb)[0]))  # type: ignore[arg-type]

    if not pair_rs:
        return {"mean_r": float("nan"), "std_r": float("nan"), "n_pairs": 0}
    return {
        "mean_r": round(float(np.mean(pair_rs)), 4),
        "std_r": round(float(np.std(pair_rs)), 4),
        "n_pairs": len(pair_rs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="02_moa_ceiling")
    parser.add_argument(
        "--smoke", action="store_true", help="Subsample to 50 drugs for quick test"
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────
    dr = pd.read_parquet(ROOT / "data" / "processed" / "drug_response.parquet")
    print(f"Loaded drug_response: {len(dr):,} pairs, "
          f"{dr['drug_name'].nunique()} drugs, {dr['depmap_id'].nunique()} cells")

    if args.smoke:
        rng_smoke = np.random.default_rng(0)
        all_drugs = sorted(dr["drug_name"].unique())
        keep = rng_smoke.choice(all_drugs, size=min(50, len(all_drugs)), replace=False)
        dr = dr[dr["drug_name"].isin(keep)].reset_index(drop=True)
        print(f"[SMOKE] Subsampled to {len(keep)} drugs")

    # ── Build response matrix ─────────────────────────────────────────
    response_matrix, drug_names, _cell_ids = build_response_matrix(dr)
    name_to_idx = {d: i for i, d in enumerate(drug_names)}
    print(f"Response matrix: {response_matrix.shape[0]} drugs x {response_matrix.shape[1]} cells")

    # ── Load MoA annotations and group drugs ──────────────────────────
    moa = load_moa_annotations()
    drug_groups = group_drugs_by_moa(drug_names, moa)

    # Filter to groups with >= MIN_DRUGS_PER_GROUP drugs present in matrix
    drug_groups = {
        g: [d for d in members if d in name_to_idx]
        for g, members in drug_groups.items()
    }
    drug_groups = {g: m for g, m in drug_groups.items() if len(m) >= MIN_DRUGS_PER_GROUP}
    print(f"MoA groups with >= {MIN_DRUGS_PER_GROUP} drugs: {len(drug_groups)}")

    # ── Within-MoA pairwise concordance ───────────────────────────────
    total_within_pairs = 0
    per_moa: list[dict] = []
    for group, members in sorted(drug_groups.items()):
        pair_rs = pairwise_r_for_group(response_matrix, name_to_idx, members)
        if not pair_rs:
            continue
        entry = {
            "moa": group,
            "mean_r": round(float(np.mean(pair_rs)), 4),
            "std_r": round(float(np.std(pair_rs)), 4),
            "min_r": round(float(np.min(pair_rs)), 4),
            "max_r": round(float(np.max(pair_rs)), 4),
            "n_drugs": len(members),
            "n_pairs": len(pair_rs),
            "drugs": sorted(members),
        }
        per_moa.append(entry)
        total_within_pairs += len(pair_rs)
        print(f"  {group}: mean_r={entry['mean_r']:.3f}  "
              f"n_drugs={entry['n_drugs']}  n_pairs={entry['n_pairs']}")

    # Sort by mean_r descending
    per_moa.sort(key=lambda x: x["mean_r"], reverse=True)

    # ── Random-pair baseline ──────────────────────────────────────────
    n_random_pairs = max(total_within_pairs, 500)
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"\nComputing random baseline ({n_random_pairs} pairs)...")
    random_baseline = compute_random_baseline(
        response_matrix, n_random_pairs, rng
    )
    print(f"  Random baseline: mean_r={random_baseline['mean_r']:.4f}  "
          f"std_r={random_baseline['std_r']:.4f}  n_pairs={random_baseline['n_pairs']}")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "random_baseline": random_baseline,
        "per_moa": per_moa,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  MoA groups analyzed: {len(per_moa)}")
    print(f"  Total within-MoA pairs: {total_within_pairs}")
    print(f"  Random baseline mean_r: {random_baseline['mean_r']:.4f}")
    if per_moa:
        above = sum(1 for m in per_moa if m["mean_r"] > random_baseline["mean_r"])
        print(f"  MoA groups above random: {above}/{len(per_moa)}")
        print(f"  Top MoA: {per_moa[0]['moa']} (mean_r={per_moa[0]['mean_r']:.3f})")
        print(f"  Bottom MoA: {per_moa[-1]['moa']} (mean_r={per_moa[-1]['mean_r']:.3f})")


if __name__ == "__main__":
    main()
