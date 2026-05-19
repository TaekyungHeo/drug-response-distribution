# PLAN: Ranking Loss — Is the Ceiling MSE-Loss-Specific?

## Hypothesis

Ridge is trained with MSE loss but evaluated with per-drug Pearson r, a ranking metric.
These objectives are mismatched: MSE minimizes absolute prediction error, not
within-drug cell ranking. Drug-standardized Ridge (within-drug z-score normalization)
directly targets within-drug ranking because standardizing each drug's IC50 values
before training makes MSE equivalent to maximizing within-drug correlation.

Prediction (null): drug-standardized r ≈ raw Ridge r (Δ ≤ 0.01), confirming that
Ridge's MSE loss already implicitly maximizes within-drug correlation (since cell
features are the same for all drugs, the ranking is drug-independent).
Prediction (alternative): standardized r > 0.641 → revise to "MSE-loss ceiling."

Mechanistic reasoning for null: a no-drug-feature Ridge learns one linear function
of cell features that is shared across all drugs. The within-drug ranking
(Pearson r) of this function is the same regardless of whether targets are
standardized, because the rankings are ordinal and the cell features don't change.

## Design

- **Model**: Ridge(α=1.0), RNA PCA(550) + mut PCA(200), PASO 10-fold drug-blind CV
- **Metric**: mean per-drug Pearson r

| Condition | Training targets | Expected Δ |
|-----------|-----------------|------------|
| `ridge_mse` | raw ln_ic50 | 0 (reference) |
| `ridge_rank` | drug-standardized ln_ic50 (within-drug z-scores) | ≈0 |
| `ridge_rank_alpha_sweep` | standardized, α ∈ {0.1, 1.0, 10.0} | ≈0 |

Drug standardization: for each training drug d, z-score = (ln_ic50 − mean_d) / std_d.
Drugs with std < 0.01 ln-unit are kept with std=0.01 (numerical stability).
At test time, predictions are within-drug rankings; per-drug r is scale-invariant.

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/04_methodological_robustness/03_ranking_loss/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/04_methodological_robustness/03_ranking_loss/jobs/sbatch.sh
```

Expected runtime: ~15 min (Ridge, CPU, 10 folds × 4 conditions)

## Validation checks

- `ridge_mse` per-drug r ≈ 0.631 ± 0.023 (matches 01_split_ceilings)
- `ridge_rank` Δ vs mse: expected |Δ| < 0.005 (linear model, same ranking)
- Drug train means/stds: log distribution (expected std ≈ 0.5–1.0 ln units)

## Output

`report/data/results.json`:
```json
{
  "summary": {
    "ridge_mse":   {"per_drug_r_mean": float, "per_drug_r_std": float, "delta": 0.0},
    "ridge_rank":  {"per_drug_r_mean": float, "per_drug_r_std": float, "delta": float},
    "ridge_rank_01": {"per_drug_r_mean": float, "per_drug_r_std": float, "delta": float},
    "ridge_rank_10": {"per_drug_r_mean": float, "per_drug_r_std": float, "delta": float}
  },
  "verdict": "string",
  "fold_results": {...}
}
```

## Dependencies

- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/cell_line_index.parquet`
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.mean_per_drug_r`
- `src/utils/paso_folds.{load_paso_pairs, load_cell_line_index}`
- `src/utils/ridge.{compress_cell, safe_fit_scaler}`

## Resources

- `--cpus-per-task=8`
- `--mem=32G`
- `--time=1:00:00`
- No GPU needed
