# 02_moa_weighted — MoA-weighted training (soft within-MoA)

## Research question

Can upweighting same-MoA drugs during training capture most of the within-MoA
benefit without discarding all other drugs?

Strict within-MoA training (01_within_moa) discards all out-of-class drugs. This
is wasteful when the MoA class is small and fragile when the class boundary is
noisy. A soft version that trains on all drugs but gives higher weight to same-MoA
drugs should interpolate between the all-drug baseline and the strict within-MoA
result.

## Hypothesis

**Null**: Upweighting same-MoA drugs does not improve per-drug r beyond the
all-drug baseline (0.644 cell-mean oracle ceiling). Ridge regression is insensitive
to sample weights at this scale.

**Alternative**: Increasing same-MoA weight monotonically improves per-drug r for
hard classes (ERK MAPK, EGFR), with diminishing returns. At 20x weight, expected
per-drug r ~0.675 -- better than all-drug (0.644) but below strict within-MoA
(~0.725 for ERK MAPK), as out-of-class drugs dilute the signal.

## Design

**Data**: GDSC2, 233 drugs, 687 cell lines, PASO 10-fold drug-blind CV.
MoA annotations: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`.

**Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200), with sample weights.

**Procedure**:
1. For each test drug in each fold:
   a. Assign weight = W to all training samples from drugs in the same MoA class.
   b. Assign weight = 1 to all other training samples.
   c. Fit Ridge with sample_weights.
   d. Predict and compute per-drug Pearson r.
2. Sweep W in {1, 2, 5, 10, 20}.
   - W=1 is the all-drug baseline (sanity check).
3. Report per-drug r as a function of W, broken down by MoA class.

**Metric**: Per-drug Pearson r, macro-averaged within each MoA class and overall.

## Validation checks

- W=1 must reproduce the all-drug baseline per-drug r exactly (same model, same data).
- Per-drug r should be monotonically non-decreasing with W for hard classes.
  Non-monotonicity at low W would suggest noise, not signal.
- Easy classes (Mitosis, Cell cycle) should be flat or slightly declining with W
  (overweighting a narrow class for already-easy drugs may hurt).
- At W=infinity (all weight on same-MoA), results should converge toward
  01_within_moa strict results.

## Output

**`report/data/results.json`** schema:
```json
{
  "weights": [1, 2, 5, 10, 20],
  "overall": [
    {
      "weight": 1,
      "mean_per_drug_r": 0.38,
      "n_drugs": 233
    }
  ],
  "per_moa": [
    {
      "moa": "ERK MAPK signaling",
      "weight": 1,
      "mean_r": 0.326,
      "n_drugs": 12
    }
  ]
}
```

**`report/data/weight_sweep.csv`**: flat table (moa, weight, mean_r, std_r, n_drugs).

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- MoA: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
- Baseline comparison: `01_within_moa/report/data/results.json`
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`

## Resources

CPU only, <1 h (5 weight values x 10 folds x 233 drugs), --mem=32G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/02_training_distribution/02_moa_weighted/jobs/run.py
```

## Downstream use

If the weight curve shows smooth improvement, this is the practical deployment
method: it works for all drugs (including those with small MoA classes) and does
not require discarding data. Feeds into `05_combinations/` as a more robust
variant of within-MoA training.
