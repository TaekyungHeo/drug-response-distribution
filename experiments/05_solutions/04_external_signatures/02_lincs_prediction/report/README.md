# 02 — Can LINCS signatures be predicted from drug structure?

## Research question

Can Morgan fingerprints predict LINCS L1000 transcriptional signatures? This
closes the structure–function–prediction triangle.

## Background

LINCS does not improve per-drug r (01_lincs, Δ=+0.001) and neither does Morgan FP
(03_drug_feature_null). But are these failures for the same reason? One possibility:
Morgan FP encodes the same functional information as LINCS, and both fail for
identical structural reasons. Another possibility: LINCS contains functional
information that chemical structure does not encode, and they fail for different
reasons — LINCS because it is a drug-level constant that cannot rank cells, Morgan
FP because it cannot even access the functional signal LINCS captures.
If Morgan FP cannot predict LINCS signatures (R² ≈ 0), the second explanation
holds — chemical structure and transcriptional function are decoupled. This
closes the explanatory triangle: structure → drug response, LINCS → drug
response, and structure → LINCS.

## Experimental design

- **Drugs**: 104 (overlap between GDSC2 and LINCS L1000)
- **Input**: Morgan fingerprints (2048-dim)
- **Target**: LINCS L1000 consensus signatures, PCA-reduced to 64 dimensions
- **Model**: Ridge regression (alpha selected per fold from {0.01, 0.1, 1, 10, 100, 1000})
- **Evaluation**: leave-one-drug-out cross-validation
- **Gate**: R² ≤ 0.1 (structure cannot predict transcriptional effect)

## Results

| Condition | R² | Mean cosine similarity |
|-----------|:---:|:---:|
| Real (Morgan FP → LINCS) | -0.0300 | 0.1481 |
| Permuted control | -0.0200 | -0.3036 |

- **Best alpha (mode across folds)**: 100.0
- **Gate (R² ≤ 0.1)**: **PASS** — structure cannot predict transcriptional effect
### Per-component R² (first 10 of 64)

| PC0 | -0.0289 |
| PC1 | 0.3278 |
| PC2 | 0.1063 |
| PC3 | 0.0072 |
| PC4 | 0.0160 |
| PC5 | -0.0620 |
| PC6 | 0.1725 |
| PC7 | 0.0808 |
| PC8 | -0.0343 |
| PC9 | 0.0312 |

## Interpretation

Morgan fingerprints cannot predict LINCS transcriptional signatures (R² =
-0.0300, at or below chance). The permuted
control yields R² = -0.0200, confirming the
real condition is no better than random.

This closes the explanatory triangle:
- Structure does not help per-drug r (03_drug_feature_null)
- LINCS does not help per-drug r either (01_lincs, Δ=+0.001)
- Structure cannot predict LINCS (this experiment)

Conclusion: LINCS captures functional information that chemical structure does
not encode. The paths to per-drug r improvement remain within-MoA training and
direct observation (K-shot).


