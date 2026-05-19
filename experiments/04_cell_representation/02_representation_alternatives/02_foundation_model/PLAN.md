# PLAN: Foundation Model — Does scFoundation Break the Per-Drug r Ceiling?

## Hypothesis

scFoundation (768-dim embeddings pretrained on 50M single cells) provides a richer,
non-linear cell representation than bulk RNA PCA. If the per-drug r ceiling is
representation-limited, scFoundation should improve over RNA PCA(550).

Prediction (null): scFoundation per-drug r ≈ RNA PCA per-drug r (Δ≈0), and
fold-level Pearson(preds_A, preds_B) ≈ 1.000 — the two representations learn
the same predictive function, confirming the ceiling is fundamental to the drug-blind
problem, not a failure of the cell representation.

## Design

- **Model**: Ridge(α=1.0), no drug features
- **Cell pool**: RNA ∩ mutations ∩ scFoundation coverage (~561 cells)
- **Splits**: PASO 10-fold drug-blind CV (restricted to scFoundation cells)
- **Primary metric**: mean per-drug Pearson r

| Condition | Cell features | Expected Δ vs A |
|-----------|--------------|----------------|
| `A_rna_mut_pca` | RNA PCA(550) + mut PCA(200) | 0 (reference) |
| `B_scfoundation` | scFoundation 768-dim | ≈0 |
| `C_concat` | RNA PCA(550) + mut PCA(200) + scFoundation | ≈0 |

**Secondary metric**: fold-level Pearson(preds_A, preds_B) — expected ≈1.000.

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/02_representation_alternatives/04_foundation_model/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/02_representation_alternatives/04_foundation_model/jobs/sbatch.sh
```

Expected runtime: ~15 min (10 folds, 3 conditions, CPU Ridge)

## Validation checks

- `A_rna_mut_pca` per-drug r ≈ 0.636 (slightly higher than full set due to smaller, higher-quality cell subset)
- `B_scfoundation` per-drug r ≈ 0.636 (Δ < 0.005)
- `C_concat` per-drug r ≈ 0.636 (no improvement from concatenation)
- Mean Pearson(preds_A, preds_B) ≈ 1.000

## Output

`report/data/results.json` — schema:
```json
{
  "summary": {
    "A_rna_mut_pca": {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_A": 0.0},
    "B_scfoundation": {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_A": float},
    "C_concat":       {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_A": float}
  },
  "pred_corr_ab": [float, ...],
  "mean_pred_corr_ab": float,
  "verdict": "string",
  "fold_results": { ... }
}
```

## Dependencies

- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/cell_line_index.parquet`
- `data/external/scFoundation/50M-0.1B-res_embedding.npy` (561×768 float32)
- `data/external/scFoundation/cancer_cell_line.info` (561 DepMap IDs)
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.mean_per_drug_r`
- `src/utils/paso_folds.{load_paso_pairs, load_cell_line_index}`
- `src/utils/ridge.{compress_cell, safe_fit_scaler}`

## Resources

- `--cpus-per-task=8`
- `--mem=32G` (scFoundation embeddings 561×768×4B ≈ 1.7MB; total peak <500MB)
- `--time=2:00:00`
- No GPU needed
