# 01_response_matching — K-shot response matching

## Research question

Can observing a new drug's IC50 on K cells break the cell-mean oracle ceiling
(per-drug r = 0.644)?

With zero observations (K=0), the best per-drug prediction is the cell-mean prior
(r ~ 0.637): predict each cell's average response across all training drugs.
A few direct observations should anchor the drug's potency and selectivity,
enabling prediction above what cell features alone can achieve.

## Hypothesis

**Null**: K-shot response matching yields per-drug r no higher than the cell-mean
oracle (0.644) for any K in {1, 3, 5, 10, 20, 50}. The K observations are too
sparse to identify meaningful drug neighbors.

**Alternative**: Response matching exceeds 0.644 once K is large enough to
reliably identify similar training drugs. Expected crossover at K ~ 10-15.

Expected magnitudes:
- K=0: per-drug r ~ 0.637 (cell-mean prior, no observations)
- K=5: per-drug r ~ 0.66 (marginal gain, noisy matching)
- K=10: per-drug r ~ 0.68 (crossover past oracle ceiling)
- K=20: per-drug r ~ 0.70
- K=50: per-drug r ~ 0.713

Measurement ceiling (r_yy) = 0.754 bounds all methods.

## Design

**Data**: GDSC2, 233 drugs, 687 cell lines, PASO 10-fold drug-blind CV.

**Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200).

**Procedure**:
1. For each fold, for each held-out (test) drug d:
   a. Sample K cells uniformly at random from the test drug's observed cells.
      These are the "anchor" cells with known IC50 for drug d.
   b. Construct the response vector v_d of length K (observed IC50 on anchor cells).
   c. For each training drug t, extract its IC50 on the same K anchor cells.
      Compute similarity: Pearson correlation between v_d and v_t (on the K cells).
   d. Compute matching prediction: weighted average of training drugs' full
      predicted profiles (Ridge predictions), with weights proportional to
      softmax(similarity / temperature).
   e. Blend: final_pred = (1 - w) * cell_mean_prior + w * matching_pred,
      where w is optimized per K via inner CV.
   f. Evaluate per-drug Pearson r on the remaining (non-anchor) cells.
2. Repeat step 1 with 10 random anchor-set draws per drug to reduce variance.
3. Average per-drug r across folds and draws.

**K sweep**: 0, 1, 3, 5, 10, 20, 50.

**Blending**: For each K, sweep w in {0.0, 0.1, 0.2, ..., 1.0}. Report both
the oracle-optimal w and inner-CV-selected w.

**Control — permuted responses**: Permute the K observed IC50 values (breaking
the cell-drug correspondence) before computing similarity. This should collapse
to cell-mean performance at all K, confirming that gains come from response
matching and not from the blending machinery.

**Metric**: Per-drug Pearson r, macro-averaged across drugs, evaluated on
non-anchor cells only.

## Validation checks

- K=0 must reproduce cell-mean prior per-drug r ~ 0.637 (within 0.01).
- Permuted control must remain at ~ 0.637 for all K.
- Per-drug r must be computed on non-anchor cells only (no data leakage from
  the K observed cells into the evaluation).
- For K=50, at least 200 cells remain for evaluation per drug (GDSC2 has ~687).
- Blending weight w should increase monotonically with K (more observations
  -> more trust in matching). Non-monotonic w signals instability.
- Per-drug r must not exceed measurement ceiling (0.754) for any individual K.

## Output

**`report/data/results.json`** schema:
```json
{
  "overall": {
    "cell_mean_prior_r": 0.637,
    "cell_mean_oracle_r": 0.644,
    "measurement_ceiling": 0.754
  },
  "k_curve": [
    {
      "k": 10,
      "mean_r": 0.68,
      "std_r": 0.03,
      "optimal_w": 0.45,
      "cv_w": 0.40,
      "permuted_r": 0.637,
      "n_drugs": 233
    }
  ],
  "per_drug": [
    {
      "drug": "Trametinib",
      "drug_id": 1372,
      "k": 10,
      "mean_r": 0.72,
      "n_folds": 10,
      "n_draws": 10
    }
  ]
}
```

**`report/data/k_curve.csv`**: flat table (k, mean_r, std_r, optimal_w, cv_w, permuted_r).

**`report/data/per_drug_by_k.csv`**: flat table (drug, drug_id, k, mean_r, std_r).

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`
- Baseline: cell-mean oracle r = 0.644 (from 05_ceilings experiments)

## Resources

CPU only, <30 min, --mem=32G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/03_few_shot/01_response_matching/jobs/run.py
```

## Downstream use

The K-curve establishes the marginal value of each additional observation.
If response matching breaks 0.644, it feeds into:
- `02_active_selection` (which cells to observe for maximum gain)
- `05_combinations/01_moa_x_kshot` (combining within-MoA training with K-shot)
