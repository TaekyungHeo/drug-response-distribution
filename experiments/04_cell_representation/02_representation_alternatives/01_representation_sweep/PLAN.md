# PLAN: Representation Sweep — Do Richer Cell Encodings Break the Ceiling?

## Hypothesis

The drug-blind per-drug r ceiling (r≈0.631) is saturated by RNA+mut PCA features.
All encodings with ≥500 dimensions should converge to the same value (Δ≈0 vs baseline)
with fold-level prediction correlations near 1.000, predicting that foundation models
will also fail to break the ceiling.

## Design

- **Model**: Ridge, no drug features, PASO 10-fold drug-blind CV
- **Cell pool**: RNA ∩ mutations ∩ pathway_features (~680 cells; same for all conditions)
- **Primary metric**: mean per-drug Pearson r

| Condition | Cell features | α | Expected Δ |
|-----------|--------------|---|------------|
| `baseline` | RNA PCA(550) + mut PCA(200) | 1.0 | 0 (reference) |
| `pca_1500` | RNA PCA(1500) + mut PCA(200) | 10.0 | ≈0 |
| `pca_max` | RNA PCA(n_train−1) + mut PCA(n_train−1) | 10.0 | ≈0 |
| `full_rna` | raw RNA (19k) + raw mut | 100.0 | ≈0 |
| `pathway_kegg` | KEGG pathway scores (1284 features) | 1.0 | ≈0 |
| `rna_plus_path` | RNA PCA(550) + pathway features | 1.0 | ≈0 |

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/02_representation_alternatives/03_representation_sweep/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/02_representation_alternatives/03_representation_sweep/jobs/sbatch.sh
```

Expected runtime: ~30 min (10 folds × 6 conditions × 3 PCA fits each)

## Validation checks

- `baseline` per-drug r ≈ 0.631 ± 0.023
- All other conditions: |Δ vs baseline| < 0.005
- `pca_max` and `full_rna` slightly lower or equal (regularization effect expected)
- `pathway_kegg` Δ ≈ 0 (confirms pathway scores don't add independent signal)

## Output

`report/data/results.json` — schema:
```json
{
  "summary": {
    "baseline": {
      "per_drug_r_mean": float, "per_drug_r_std": float,
      "global_r_mean": float, "delta_vs_baseline": float
    },
    ...
  },
  "fold_results": {
    "baseline": [{"per_drug_r": float, "global_r": float}, ...]
  }
}
```

## Dependencies

- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/pathway_features.parquet`
- `data/processed/cell_line_index.parquet`
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.mean_per_drug_r`
- `src/utils/paso_folds.{load_paso_pairs, load_cell_line_index}`
- `src/utils/ridge.{compress_cell, safe_fit_scaler}`

## Resources

- `--cpus-per-task=8`
- `--mem=32G` (largest feature set: full_rna ~687×19k×4B = 52MB; peak <500MB)
- `--time=2:00:00`
- No GPU needed
