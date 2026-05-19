# PLAN: Nonlinear Models — Is the Ceiling Ridge-Specific?

## Hypothesis

Ridge(α=1.0) is a linear model. The drug-blind per-drug r=0.631 may be a
"Ridge ceiling" rather than a drug-blind ceiling. If XGBoost or MLP captures
nonlinear transcriptomic interactions, they could exceed r=0.631, meaning the
problem is not fundamentally hard — just representation-limited for linear models.

Prediction (null): XGBoost per-drug r ≈ Ridge r (Δ ≤ 0.01), confirming the ceiling
is structural to the drug-blind problem, not an artifact of linear modeling.
Prediction (alternative): XGBoost per-drug r > 0.641 → revise to "Ridge ceiling."

## Design

- **Data**: RNA PCA(550) + mut PCA(200), PASO 10-fold drug-blind CV (233 drugs, ~687 cells)
- **Metric**: mean per-drug Pearson r
- **Models**:

| Condition | Model | Hyperparameters | Expected Δ vs Ridge |
|-----------|-------|----------------|---------------------|
| `ridge` | Ridge(α=1.0) | — | 0 (reference) |
| `xgboost` | XGBRegressor | n_est=300, max_depth=6, lr=0.05, sub=0.8 | ≈0 |
| `mlp` | MLPRegressor | hidden=(512,256), relu, max_iter=500, early_stopping | ≈0 |

Note: same cell features as Ridge baseline (RNA PCA 550 + mut PCA 200).
PCA is fit on training cells only (same compress_cell pipeline).

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/04_methodological_robustness/01_nonlinear_models/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/04_methodological_robustness/01_nonlinear_models/jobs/sbatch.sh
```

Expected runtime: ~45 min (10 folds × 3 models; XGBoost ~3× Ridge time)

## Validation checks

- `ridge` per-drug r ≈ 0.631 ± 0.023 (matches 01_split_ceilings)
- `xgboost` Δ vs ridge: if |Δ| ≤ 0.01 → "not Ridge-limited"
- `mlp` Δ vs ridge: similar
- Fold-level Pearson(ridge_preds, xgboost_preds) ≈ 1.000 if both learn the same function

## Output

`report/data/results.json`:
```json
{
  "summary": {
    "ridge":   {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_ridge": 0.0},
    "xgboost": {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_ridge": float},
    "mlp":     {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_ridge": float}
  },
  "pred_corr": {"ridge_xgboost": [float, ...], "ridge_mlp": [float, ...]},
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
- `xgboost` (installed via uv add xgboost)
- `sklearn.neural_network.MLPRegressor`

## Resources

- `--cpus-per-task=8`
- `--mem=32G`
- `--time=2:00:00`
- No GPU needed (XGBoost CPU, MLP CPU)
