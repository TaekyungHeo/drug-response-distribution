# PLAN: Oracle Bounds

## What this experiment answers

Before testing whether drug representations help, we need to know what ceiling they could
realistically reach. Two bounds are relevant:

**Bound 1 — Between-drug oracle (global r)**
A predictor that outputs each test drug's true mean IC₅₀ for every cell line achieves the
maximum global r attainable by perfectly predicting between-drug potency differences. This
bound is achievable only by cheating (using test drug means); it sets the ceiling for the
global-r metric. Any model reaching global r well below this has not captured the between-drug
signal — but any model with global r near this value may simply be predicting drug means, not
cell-line sensitivity.

**Bound 2 — Profile concordance (per-drug r)**
A model that predicts test-drug sensitivity by copying the profile of the most similar training
drug is bounded by the within-class profile concordance: the mean per-drug r among drugs that
share a mechanism of action. This measures the maximum per-drug r achievable by drug-similarity
transfer alone — i.e., predicting a test drug's cell-line ranking entirely from the most similar
training drug's profile, without using any cell features.

Note: profile concordance ≈ 0.52 is **not** the ceiling for per-drug r overall. Cell features
alone already achieve per-drug r ≈ 0.631, which exceeds 0.52. These two numbers measure
different things: 0.52 is the ceiling for the *drug-similarity transfer mechanism specifically*,
while 0.631 reflects the cell-feature signal. Profile concordance upper-bounds the marginal
contribution that drug features can make *via similarity transfer* — not the contribution that
cell features make independently.

Δ=+0.003 ≈ 0.6% of 0.52 means drug features extract essentially none of the similarity-transfer
signal that is theoretically available.

## Design

### Bound 1: Drug-mean oracle

- Dataset: GDSC2, 233 drugs with SMILES (same as main ablation)
- Protocol: for each test fold in PASO 5-fold drug-blind CV, predict every (cell, drug) pair
  in the test set as the true mean IC₅₀ of that drug (computed from test pairs)
- Metric: global Pearson r (per-fold mean ± std)
- This oracle requires test-drug means → not achievable in real drug-blind evaluation;
  its purpose is to show how much of global r is explained by between-drug variation alone

### Bound 2: Within-class profile concordance

- Dataset: GDSC2 response matrix, GDSC2 pathway annotations (from GDSC2 metadata)
- Protocol: for each pair of drugs annotated to the same pathway, compute per-drug Pearson r
  between their cell-line IC₅₀ profiles (on shared cell lines, ≥5 cells required)
- Report: mean ± std across all within-class pairs; also broken down by pathway
- This is an upper bound for the drug-feature similarity transfer mechanism: even a perfect drug
  similarity oracle that always retrieves the most similar training drug cannot exceed this value
  for within-drug cell ranking. This bound does not apply to cell-feature-based prediction.

**Secondary concordance estimate (Tanimoto-based)**:
GDSC2 pathway annotations are coarse (many drugs per pathway); 0.52 may underestimate the
true ceiling. Compute a second concordance using structural similarity: for each drug, find
its nearest neighbor by Tanimoto similarity (Morgan FP, radius=2) and compute per-drug r
between their profiles. Report: mean ± std across all drug pairs with Tanimoto ≥ 0.7.
If this estimate is substantially higher than 0.52, the pathway-based bound is conservative
and the 0.6% framing becomes even stronger. If lower, the 0.52 bound may be overestimating
the drug-similarity signal.

## Expected results

These two bounds measure different things and must not be compared directly:

| Bound | Metric | Expected value | Interpretation |
|-------|--------|---------------|----------------|
| Drug-mean oracle | **global r** | ≈ 0.79 (expected from prior work; will be measured) | Ceiling for between-drug scale prediction; 63% of global r explained by drug identity alone |
| Profile concordance | **per-drug r** | ≈ 0.52 (mean across pathways) | Marginal ceiling for drug-similarity transfer mechanism; Δ=+0.003 ≈ 0.6% of this ceiling |

The oracle bound (global r ≈ 0.79) is relevant to the solutions section, not the null result.
The profile concordance (per-drug r ≈ 0.52) upper-bounds the drug-similarity transfer mechanism —
it is the marginal ceiling for what drug features can add via similarity transfer, not the
ceiling for per-drug r overall (cell features already achieve 0.631 independently).

## Prerequisites

- `data/processed/drug_response.parquet` (GDSC2 IC₅₀ matrix)
- PASO 5-fold drug-blind splits (`external/PASO/data/10_fold_data/drug_blind/`)
- GDSC2 drug annotations (pathway / MoA labels)

## Pre-flight checks

Before running, verify:
1. `data/processed/drug_response.parquet` present; spot-check shape ≥ (600 cells, 200 drugs)
2. PASO fold CSV files present for folds 0–4 (drug-blind)
3. GDSC2 drug annotation file present with pathway/MoA columns
4. SMILES file present for Tanimoto concordance computation (requires Morgan FP computation)
5. At least 10 drugs with ≥ 2 same-pathway partners (required for within-class concordance)

## How to run

```bash
sbatch experiments/03_drug_feature_null/01_oracle_bounds/jobs/sbatch.sh
```

Expected runtime: < 5 min (no training). SLURM: `--mem=8G`, no GPU, `--time=0:30:00`.

## Output

```
report/data/metrics.json   — oracle_global_r, profile_concordance_mean/std, by_pathway breakdown
```
