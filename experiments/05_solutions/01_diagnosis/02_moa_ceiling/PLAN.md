# 02_moa_ceiling — Within-MoA biological ceiling

## Research question

What is the within-MoA biological ceiling for per-drug r?

If drugs in the same MoA class have similar response profiles across cell lines,
then a model trained only on same-MoA drugs can borrow useful signal. The
profile concordance between same-MoA drugs sets the ceiling for this borrowing.

## Hypothesis

Drugs sharing a MoA target similar (but not identical) cellular vulnerabilities.
Expected within-MoA pairwise r: ~0.5-0.7, varying by class. Classes with high
concordance (e.g., Mitosis) have a high ceiling for within-MoA training. Classes
with low concordance (e.g., Other, Unclassified) gain nothing from MoA-stratified
training because same-class drugs are not more informative than random drugs.

## Design

**Data**: GDSC2 drug response matrix, 233 drugs, 687 cell lines.
MoA annotations: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`.

**Procedure**:
1. Build a drug-by-cell response matrix (drugs as rows, cells as columns).
2. For each MoA class with ≥3 drugs:
   a. Extract the submatrix of drugs in that class.
   b. Compute all pairwise Pearson r between drug response profiles
      (across shared cell lines, ≥20 shared cells required).
   c. Report: mean pairwise r, std, min, max, n_drugs, n_pairs.
3. As a control, compute pairwise r for random drug pairs (same n_pairs,
   sampled across all drugs regardless of MoA). This estimates the baseline
   concordance from cell-identity effects alone.

**Metric**: Pairwise per-profile Pearson r (each drug's IC50 vector across cells).

**Interpretation**:
- High within-MoA r (>> random baseline): MoA carries real signal; within-MoA
  training has a high ceiling.
- Within-MoA r ≈ random baseline: MoA label does not capture shared response
  structure; within-MoA training is unlikely to help for that class.

## Validation checks

- Random-pair baseline r should be positive (cell-identity drives some shared
  variance) but lower than most within-MoA values.
- Symmetric: r(drug_A, drug_B) = r(drug_B, drug_A). Verify no duplicates in pairs.
- MoA classes with <3 drugs excluded (≥3 needed for ≥3 pairs).
- Sanity: Mitosis drugs should have high concordance (known biology).

## Output

**`report/data/results.json`** schema:
```json
{
  "random_baseline": {
    "mean_r": 0.35,
    "std_r": 0.15,
    "n_pairs": 500
  },
  "per_moa": [
    {
      "moa": "Mitosis",
      "mean_r": 0.65,
      "std_r": 0.10,
      "min_r": 0.48,
      "max_r": 0.82,
      "n_drugs": 8,
      "n_pairs": 28,
      "drugs": ["Docetaxel", "..."]
    }
  ]
}
```

**`report/data/moa_ceiling.csv`**: flat table (moa, mean_r, std_r, n_drugs, n_pairs).

## Dependencies

- Data: `data/processed/drug_response.parquet`
- MoA: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
- No model code required (pure data analysis).

## Resources

CPU only, <5 min, --mem=16G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/01_diagnosis/02_moa_ceiling/jobs/run.py
```

## Downstream use

Compare with `01_moa_performance` results: if a MoA class has high ceiling (high
profile concordance) but low observed per-drug r under all-drug training, that
class is the best candidate for improvement via within-MoA training
(`02_training_distribution/01_within_moa`). The gap between ceiling and observed
performance quantifies the opportunity.
