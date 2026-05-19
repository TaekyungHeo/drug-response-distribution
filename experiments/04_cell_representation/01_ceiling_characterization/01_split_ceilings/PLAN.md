# PLAN: Split Ceilings — Drug-Blind vs Cell-Blind Per-Drug r

## Hypothesis

Drug-blind (unseen drugs) and cell-blind (unseen cells) impose different generalization
demands. Prediction: cell-blind per-drug r < drug-blind per-drug r (r≈0.461 vs r≈0.631).
The drug-mean cheat predictor should reproduce Sealfon-style global r inflation (~0.79)
while scoring near-zero on per-drug r, validating per-drug r as the correct metric.

## Design

- **Model**: Ridge(α=1.0), no drug features, RNA PCA(550) + mut PCA(200)
- **Drug-blind**: PASO 10-fold drug-blind CV (233 drugs, ~687 cells), per-drug Pearson r
- **Cell-blind**: 5-fold CV on cells (same 233 PASO drugs, ~687 cells), per-drug Pearson r
- **Cheat predictor**: drug-mean IC50 from fold-0 training data — no test leakage
- **Primary metric**: mean per-drug Pearson r (averaged over drugs, then over folds)

| Condition | Split | Expected per-drug r |
|-----------|-------|-------------------|
| drug_blind | PASO 10-fold drug-blind | 0.631 ± 0.023 |
| cell_blind | 5-fold cell-blind | 0.461 ± 0.027 |
| cheat_predictor | fold-0 cell-blind | per-drug r ≈ 0; global r ≈ 0.791 |

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/01_ceiling_characterization/01_split_ceilings/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/01_ceiling_characterization/01_split_ceilings/jobs/sbatch.sh
```

Expected runtime: ~10 min (Ridge, CPU, 10+5 folds)

## Validation checks

- Drug-blind per-drug r ≈ 0.631 ± 0.023 (PASO 10-fold)
- Cell-blind per-drug r ≈ 0.461 ± 0.027 (5-fold on cells)
- Gap: cell-blind − drug-blind ≈ −0.170
- Cheat predictor: global r ≈ 0.791, per-drug r ≈ 0 (constant per drug)
- Drug-mean cheat: validates Sealfon-style global r inflation

## Output

`report/data/results.json` — schema:
```json
{
  "drug_blind": {
    "folds": [{"per_drug_r": float, "global_r": float, "n_drugs": int}, ...],
    "per_drug_r_mean": float, "per_drug_r_std": float, "global_r_mean": float
  },
  "cell_blind": { ... same structure ... },
  "cheat_predictor": {"global_r": float, "per_drug_r": float}
}
```

## Dependencies

- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/drug_response.parquet` (cell-blind DR pairs)
- `data/processed/cell_line_index.parquet`
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.mean_per_drug_r`
- `src/utils/paso_folds.{load_paso_pairs, load_cell_line_index}`
- `src/utils/ridge.{compress_cell, safe_fit_scaler}`

## Resources

- `--cpus-per-task=8` (Ridge uses all cores via BLAS)
- `--mem=32G` (rna.parquet ~52MB; peak ~500MB; 32G ample)
- `--time=2:00:00`
- No GPU needed
