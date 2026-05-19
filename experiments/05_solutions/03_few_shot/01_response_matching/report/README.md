# 01 — K-shot response matching

## Research question

Can observing a new drug's IC₅₀ on K cell lines break the cell-mean oracle
ceiling (per-drug r = 0.644)?

## Background

Within-MoA training requires knowing a drug's MoA in advance. For truly novel
drugs without MoA annotation, a different approach is needed. If we observe a
new drug's IC₅₀ on K pilot cell lines, we can find the training drug whose
response profile best matches those K observations, then use that drug's full
profile as the prediction. This K-shot response matching requires no MoA
knowledge — only a small set of pilot measurements. The key question is how
many cells K are needed to reliably identify the nearest neighbor and whether
that achieves per-drug r comparable to MoA-stratified training.

## Experimental design

- **CV**: PASO 10-fold drug-blind
- **Baselines**: cell-mean prior (K=0), cell-mean oracle (perfect mean knowledge), measurement ceiling (r_yy = 0.754)
- **Method**: blend nearest-neighbor profile with cell-mean prior using weight w; w and neighbors optimized per fold
- **Permuted control** (K=50): responses shuffled across cells — tests that gains come from genuine profile matching, not the blending machinery

## Results

| K | mean r | std r | optimal w | CV w | permuted r | n_drugs |
|--:|:------:|:-----:|:---------:|:----:|:----------:|:-------:|
| 0 | 0.6450 | 0.1391 | 0.0 | 0.0 | — | 233 |
| 1 | 0.6450 | 0.1391 | 0.0 | 0.1 | — | 233 |
| 3 | 0.6450 | 0.1391 | 0.0 | 0.0 | — | 233 |
| 5 | 0.6461 | 0.1344 | 0.1 | 0.1 | — | 233 |
| 10 | 0.6533 | 0.1250 | 0.2 | 0.2 | — | 233 |
| 20 | 0.6705 | 0.1164 | 0.3 | 0.3 | — | 233 |
| 50 | 0.7007 | 0.1124 | 0.5 | 0.6 | 0.5607 | 233 |

### Crossover analysis

Response matching exceeds the cell-mean oracle ceiling (0.644) starting at
**K = 0** (mean r = 0.6450).

## Interpretation

The blend weight w indicates how much trust the model places in the K-shot
neighbor prediction versus the cell-mean prior. A monotonically increasing w
with K confirms that more observations yield more reliable matching. The
permuted control at K=50 verifies that gains come from genuine response profile
similarity, not from the blending procedure itself.

Measurement ceiling (0.754) bounds all methods.
K=0 (cell-mean prior = 0.645) is the zero-observation baseline.


