# PLAN: Chemical Split — Does Tanimoto Drug-Blind CV Give a Harder Ceiling?

## Hypothesis

PASO drug-blind uses random drug assignment to train/test. Chemically similar drugs
(same scaffold, similar mechanism) may share pharmacological signal, making random
drug-blind CV easier than it appears. A Tanimoto-distance-based split — where test
drugs are maximally dissimilar from training drugs — gives a harder, more realistic
generalization estimate.

Prediction: Tanimoto-split per-drug r < random-split r=0.631. If close (Δ ≤ 0.05),
PASO random split is not misleadingly optimistic. If large (Δ > 0.15), the reported
ceiling is inflated by chemical similarity leakage.

## Design

- **Fingerprints**: ECFP4 2048-bit Morgan fingerprints, 233 PASO drugs, sorted order
  (file: `data/processed/drug_fingerprints_233.npy`)
- **Tanimoto distance**: 1 − Tanimoto similarity (binary Jaccard)
- **Clustering**: hierarchical (Ward linkage), cut into 10 clusters → 10-fold CV
  with one cluster as test in each fold
- **Model**: Ridge(α=1.0), RNA PCA(550) + mut PCA(200)
- **Metric**: mean per-drug Pearson r

| Condition | Split type | Expected per-drug r |
|-----------|-----------|---------------------|
| `random` (reference) | PASO 10-fold drug-blind | 0.631 ± 0.023 |
| `tanimoto` | Chemical-cluster 10-fold drug-blind | < 0.631 |

Note: test set size may be unequal across folds (cluster sizes vary).
Drugs with no SMILES are excluded (coverage: 233/233 confirmed).

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/04_methodological_robustness/02_chemical_split/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/04_methodological_robustness/02_chemical_split/jobs/sbatch.sh
```

Expected runtime: ~15 min (Ridge, CPU, 10 chemical folds)

## Validation checks

- Tanimoto matrix: symmetric, diagonal=1, off-diagonal values in [0,1]
- 10 chemical clusters: log cluster sizes (expected: 10–50 drugs per cluster)
- `random` per-drug r ≈ 0.631 (sanity check, recomputed from PASO splits)
- `tanimoto` r < `random` r (harder split → lower generalization)
- If |`tanimoto` − `random`| < 0.05 → chemical similarity leakage negligible

## Output

`report/data/results.json`:
```json
{
  "random": {"per_drug_r_mean": float, "per_drug_r_std": float},
  "tanimoto": {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_random": float},
  "cluster_sizes": [int, ...],
  "verdict": "string",
  "fold_results": {"random": [...], "tanimoto": [...]}
}
```

## Dependencies

- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/drug_fingerprints_233.npy` (233×2048 ECFP4, sorted PASO drug order)
- `data/processed/drug_response.parquet`
- `data/processed/cell_line_index.parquet`
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.mean_per_drug_r`
- `src/utils/paso_folds.{load_paso_pairs, load_cell_line_index}`
- `src/utils/ridge.{compress_cell, safe_fit_scaler}`
- `scipy.cluster.hierarchy`, `scipy.spatial.distance`

## Resources

- `--cpus-per-task=8`
- `--mem=32G`
- `--time=1:00:00`
- No GPU needed
