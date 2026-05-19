# 01 — Oracle Bounds

## Research question

What are the theoretical ceilings for drug feature contributions? Before testing whether
drug representations help, we need bounds that define "how much could they possibly help?"

## Results

| Bound | Metric | Value | Interpretation |
|-------|--------|:-----:|----------------|
| Drug-mean oracle | global r | 0.818 ± 0.066 | Ceiling for between-drug scale prediction (10-fold PASO splits) |
| Tanimoto concordance | per-drug r | 0.718 ± 0.114 (n=14 pairs) | Structural-similarity transfer ceiling |
| Profile concordance | per-drug r | N/A (0 qualifying pairs) | MoA-based transfer ceiling (stringent matching found no qualifying pairs) |

Note: The paper's canonical drug-mean oracle global r = 0.837 (reported as `\DrugMeanOracleR`)
comes from a 5-fold drug-blind CV in the baselines experiment, not these 10-fold PASO splits.
The MoA-based profile concordance (0.528 mean, 22 MoA classes) reported in the paper comes
from `05_solutions/01_diagnosis/02_moa_ceiling` using GDSC2 pathway annotations directly.

### Drug-mean oracle (global r)

A predictor that outputs each test drug's true mean IC50 achieves global r = 0.818
(10-fold mean). This is an unattainable upper bound — it uses test labels. Any model
with global r near this value is simply predicting drug potency means, not cell-line
sensitivity.

Fold-level values range from 0.646 to 0.889, reflecting variance in drug composition
across folds.

### Tanimoto concordance (per-drug r)

Among 14 drug pairs with high structural similarity (Tanimoto), per-drug r between their
response profiles averages 0.718. This represents the ceiling for drug-similarity transfer:
the best a model could achieve by copying the response profile of the most structurally
similar training drug.

The drug feature delta (+0.001, from 02_representation_ablation) is 0.1% of this ceiling.
Drug features extract essentially none of the similarity-transfer signal available in the
data.

### Profile concordance (per-drug r)

No qualifying drug pairs were found for MoA-based profile concordance (0 pairs). This is
likely due to insufficient pathway annotation coverage in the GDSC2 drug set for the
stringent matching criteria used.

## Interpretation

The two bounds serve different downstream experiments:

1. **Drug-mean oracle (global r = 0.818)** contextualizes the solutions section (05_solutions):
   LINCS signatures do not improve global r on the 104-drug overlap subset (actual: −0.058;
   see 05_solutions/04_external_signatures/01_lincs). The drug-mean ceiling remains as a
   reference for between-drug scale prediction.

2. **Tanimoto concordance (per-drug r = 0.718)** contextualizes this null-result section:
   drug features achieve delta = +0.001 against a ceiling of 0.718 — the mechanism exists
   in the data but drug representations fail to access it.

## Validation checks

- Drug-mean oracle global r (0.818) > all trained models' global r: expected
- Drug-mean oracle per-drug r should be ~0 (same constant per drug): not computed (structural design)
- Tanimoto concordance based on 14 pairs: small sample, high variance — interpret with caution
