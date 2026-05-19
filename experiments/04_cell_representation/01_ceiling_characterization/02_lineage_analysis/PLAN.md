# PLAN: Lineage Analysis — Per-Lineage Per-Drug r and LINCS Coverage Bias

## Hypothesis

Two reviewer objections must be ruled out before attributing r=0.631 to genuine cell-response
correlation:

1. **Lineage confounding**: the model learns cancer type → average sensitivity, not within-lineage
   cell ranking. Prediction: per-lineage per-drug r ≥ 0.47 for all major lineages (does not
   collapse to zero), confirming within-lineage signal.

2. **LINCS selection bias**: LINCS-covered drugs (104/233) are well-studied targeted agents that
   may be inherently more predictable. Prediction: LINCS-covered vs uncovered Δ ≈ 0 in the
   no-drug-features Ridge baseline, confirming LINCS does not inflate the no-feature ceiling.

## Design

- **Model**: Ridge(α=1.0), no drug features, RNA PCA(550) + mut PCA(200)
- **Splits**: PASO 10-fold drug-blind CV (233 drugs, ~687 cells)
- **Primary metric**: mean per-drug Pearson r
- **Lineages** (CCLE tissue field): Hematologic, Lung, Skin, CNS, Breast, Colorectal, Other
- **Min cells per drug for lineage eval**: 5 (same as global metric)
- **LINCS drug set**: `data/processed/lincs_drug_index.json` (104 matched drugs)

| Analysis | Expected result |
|----------|----------------|
| Pan-cancer overall | r ≈ 0.631 ± 0.023 |
| Per-lineage (all 7) | r ≥ 0.47 (no collapse) |
| LINCS covered | r ≈ uncovered (Δ ≈ 0) |

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/01_ceiling_characterization/02_lineage_analysis/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/01_ceiling_characterization/02_lineage_analysis/jobs/sbatch.sh
```

Expected runtime: ~15 min (10 folds, 7 lineage subsets per fold)

## Validation checks

- Pan-cancer per-drug r ≈ 0.631 (matches 01_split_ceilings baseline)
- All 7 lineages: per-drug r ≥ 0.47 (no lineage scores near zero)
- LINCS-covered vs uncovered: |Δ| < 0.02
- Hematologic lineage likely has the most cells (~200); CNS / Breast likely fewest (~20-40)

## Output

`report/data/results.json` — list of fold dicts, each:
```json
{
  "fold": int,
  "overall": {"mean": float, "n_drugs": int},
  "by_lineage": {
    "Hematologic": {"mean": float, "n_drugs": int, "n_pairs": int},
    ...
  },
  "lincs_covered": {"mean": float, "n_drugs": int},
  "lincs_uncovered": {"mean": float, "n_drugs": int}
}
```

## Dependencies

- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/cell_line_index.parquet` (ccle_name for lineage extraction)
- `data/processed/lincs_drug_index.json` (LINCS matched drug list)
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.mean_per_drug_r`
- `src/utils/paso_folds.{load_paso_pairs, load_cell_line_index}`
- `src/utils/ridge.{compress_cell, safe_fit_scaler}`

## Resources

- `--cpus-per-task=8`
- `--mem=32G`
- `--time=2:00:00`
- No GPU needed
