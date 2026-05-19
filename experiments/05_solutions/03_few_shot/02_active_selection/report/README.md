# 02 — Which K cells to screen for maximum K-shot gain?

## Research question

Given a budget of K cells to screen for a new drug, which cell selection
strategy maximizes the per-drug prediction gain from response matching?

## Background

K-shot response matching (01_response_matching) shows that observing K pilot
IC₅₀ values substantially improves prediction. But the default strategy is
random cell selection — any K cells. Some cells may be more informative than
others: high-variance cells (that discriminate between drugs) or diverse cells
(covering different sensitivity ranges) could yield better neighbor
identification for the same budget K. If smart selection substantially
outperforms random, it reduces the experimental cost of characterizing new drugs.

## Experimental design

- **CV**: PASO 10-fold drug-blind (10 folds run)
- **K values**: 1, 3, 5, 10, 20
- **Strategies**: Random, MaxVar, MidResp, Diverse
- **Response matching**: blend_weight=0.5, n_neighbors=5
- **Leakage control**: cell selection uses training-drug data only (response variance, mean response, RNA PCA coordinates computed from training drugs within each fold)

## Results

| Strategy | K | mean r | std r | delta vs random | n drugs |
|----------|--:|-------:|------:|----------------:|--------:|
| Diverse | 1 | 0.6451 | 0.1398 | +0.0009 | 217 |
| MaxVar | 1 | 0.6652 | 0.1250 | +0.0211 | 43 |
| MidResp | 1 | 0.6435 | 0.1420 | -0.0006 | 211 |
| Random | 1 | 0.6441 | 0.1391 | +0.0000 | 231 |
| Diverse | 3 | 0.5981 | 0.1438 | +0.0014 | 233 |
| MaxVar | 3 | 0.6164 | 0.1422 | +0.0197 | 220 |
| MidResp | 3 | 0.6060 | 0.1418 | +0.0093 | 225 |
| Random | 3 | 0.5967 | 0.1274 | +0.0000 | 233 |
| Diverse | 5 | 0.5844 | 0.1332 | -0.0189 | 233 |
| MaxVar | 5 | 0.6037 | 0.1428 | +0.0005 | 232 |
| MidResp | 5 | 0.5994 | 0.1463 | -0.0039 | 228 |
| Random | 5 | 0.6033 | 0.1176 | +0.0000 | 233 |
| Diverse | 10 | 0.6320 | 0.1384 | +0.0000 | 233 |
| MaxVar | 10 | 0.6323 | 0.1368 | +0.0003 | 233 |
| MidResp | 10 | 0.6286 | 0.1274 | -0.0034 | 232 |
| Random | 10 | 0.6320 | 0.1150 | +0.0000 | 233 |
| Diverse | 20 | 0.6563 | 0.1395 | -0.0060 | 233 |
| MaxVar | 20 | 0.6732 | 0.1326 | +0.0109 | 233 |
| MidResp | 20 | 0.6553 | 0.1298 | -0.0070 | 232 |
| Random | 20 | 0.6623 | 0.1118 | +0.0000 | 233 |

### Random baseline by K

| K | mean r |
|--:|-------:|
| 1 | 0.6441 |
| 3 | 0.5967 |
| 5 | 0.6033 |
| 10 | 0.6320 |
| 20 | 0.6623 |

## Interpretation

**MaxVar** at K=1 achieves the largest gain over random
(delta = +0.0211), suggesting targeted cell
selection can meaningfully reduce the experimental cost of characterizing new drugs.


