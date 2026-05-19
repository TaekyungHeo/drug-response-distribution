# PLAN: Multi-Omics Addition — Does CNV/Metabolomics Help Ridge Drug-Blind?

## Hypothesis

The canonical RNA+mut baseline (per-drug r=0.631) may not be the true ceiling.
CNV and metabolomics may contain complementary information about drug sensitivity
that RNA+mut misses. RPPA was already tested in `05_proteomics_oracle/` (Δ=0),
so it is excluded here to avoid redundancy.

Null hypothesis: adding extra omics types to Ridge produces Δ ≈ 0 vs RNA+mut —
meaning the ceiling is genuinely information-limited at the RNA+mut level.

## Design

- **Model**: Ridge(α=1.0), no drug features
- **Data**: GDSC2 × available cells per condition (smaller intersection than RNA+mut alone)
- **Splits**: PASO 10-fold drug-blind CV (233 drugs)
- **Primary metric**: mean per-drug Pearson r
- **Conditions**:
  - `rna_mut`       — RNA PCA(550) + mut PCA(200)  [canonical baseline, r≈0.631]
  - `rna_mut_cnv`   — adds CNV PCA(300)
  - `rna_mut_metab` — adds metabolomics (225 features, no PCA)
  - `rna_mut_all`   — RNA+mut+CNV+metabolomics

Note: RPPA excluded — `05_proteomics_oracle/` already showed Δ=0 for RPPA.

## How to run

```bash
uv run python3 experiments/04_cell_representation/02_representation_alternatives/06_multi_omics/jobs/run.py
```

Expected runtime: ~10 min (Ridge, CPU, 10-fold)

## Validation checks

- `rna_mut` per-drug r ≈ 0.631 ± 0.023 (matches canonical baseline)
- If all conditions ≈ baseline → information ceiling reached with RNA+mut
- Cell count drop for multi-omics conditions logged (intersection may reduce n)

## Output

`report/data/results.json` — dict keyed by condition name:
```json
{
  "rna_mut": {
    "modalities": ["rna", "mutations"],
    "n_cells": int,
    "folds": [{"per_drug_r": float, "feat_dim": int}, ...],
    "per_drug_r_mean": float,
    "per_drug_r_std": float,
    "delta_vs_rna_mut": float
  },
  ...
}
```

## Dependencies

- `data/processed/{rna,mutations,cnv,metabolomics}.parquet`
- `data/processed/cell_line_index.parquet`
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.mean_per_drug_r`
- `src/utils/paso_folds.{load_paso_pairs, load_cell_line_index}`
- `src/utils/ridge.{compress_multi_omics, safe_fit_scaler}`

## Resources

- `--cpus-per-task=8`
- `--mem=48G` (CNV + metabolomics intersection; conservative given cell-count reduction)
- `--time=2:00:00`
- No GPU needed
