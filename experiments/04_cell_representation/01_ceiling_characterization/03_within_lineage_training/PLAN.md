# PLAN: Within-Lineage Training — Does Cross-Lineage Variance Inflate Per-Drug r?

## Hypothesis

The pan-cancer per-drug r=0.631 may be inflated because the model learns
"hematologic cells are generally more/less sensitive than lung cells" rather than
within-lineage cell-to-cell variation. If cross-lineage mean differences drive the
score, training on a single lineage only should give dramatically lower r.

Prediction (null): within-lineage per-drug r ≈ pan-cancer per-lineage r (Δ ≤ 0.05).
Prediction (alternative): within-lineage r << pan-cancer lineage r (Δ > 0.10),
meaning the pan-cancer model exploited cross-lineage variance.

Expected results under null:
- Hematologic within-lineage r ≈ 0.58 ± 0.05
- Lung within-lineage r ≈ 0.52 ± 0.05
- Other lineages: r ≥ 0.40 (drops from lower cell counts but doesn't collapse)

## Design

- **Model**: Ridge(α=1.0), RNA PCA(550) + mut PCA(200)
- **Splits**: PASO 10-fold drug-blind CV; each fold restricts train AND test to a single lineage
- **Metric**: mean per-drug Pearson r (min_cells=5)
- **Lineages**: Hematologic, Lung, Skin, CNS, Breast, Colorectal
- **Min train cells**: 20 (skip lineage-fold if fewer training cells than this)

| Condition | Train cells | Test cells | Expected r |
|-----------|-------------|------------|------------|
| Pan-cancer (reference) | all lineages | all lineages | 0.631 |
| Hematologic-only | ~170 Hem cells | ~30 Hem cells | ≈ pan-cancer Hem r |
| Lung-only | ~80 Lung cells | ~14 Lung cells | ≈ pan-cancer Lung r |
| Skin-only | ~70 Skin cells | ~13 Skin cells | ≈ pan-cancer Skin r |
| CNS-only | ~30 CNS cells | ~5 CNS cells | ≈ pan-cancer CNS r |
| Breast-only | ~30 Breast cells | ~5 Breast cells | ≈ pan-cancer Breast r |
| Colorectal-only | ~50 CRC cells | ~9 CRC cells | ≈ pan-cancer CRC r |

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/01_ceiling_characterization/03_within_lineage_training/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/01_ceiling_characterization/03_within_lineage_training/jobs/sbatch.sh
```

Expected runtime: ~20 min (6 lineages × 10 folds, Ridge CPU)

## Validation checks

- Pan-cancer overall r ≈ 0.631 (sanity check against 01_split_ceilings)
- Hematologic within-lineage r: valid (Hem has most cells, ~200)
- CNS/Breast: fewer folds valid (many may be skipped due to min_cells=5 evaluation threshold)
- |within_lineage_r − pancancer_lineage_r| ≤ 0.05 across all lineages → null confirmed

## Output

`report/data/results.json`:
```json
{
  "pancancer_overall": {"mean": float, "std": float},
  "by_lineage": {
    "Hematologic": {
      "within_lineage_folds": [{"per_drug_r": float, "n_train_cells": int, "n_test_cells": int, "n_drugs": int}, ...],
      "within_lineage_mean": float,
      "within_lineage_std": float,
      "n_valid_folds": int
    },
    ...
  }
}
```

## Dependencies

- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/cell_line_index.parquet` (ccle_name for lineage)
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.mean_per_drug_r`
- `src/utils/paso_folds.{load_paso_pairs, load_cell_line_index}`
- `src/utils/ridge.{compress_cell, safe_fit_scaler}`

## Resources

- `--cpus-per-task=8`
- `--mem=32G`
- `--time=2:00:00`
- No GPU needed
