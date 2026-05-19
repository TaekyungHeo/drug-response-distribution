# 01 — Representation sweep: all ≥500-dim cell representations converge

## Research question

Does the choice of cell-line representation matter? We compare 6 representations
varying in dimensionality and biological structure.

## Background

The Ridge baseline uses RNA PCA(550) + mutation PCA(200). This is a practical
default, not necessarily the optimal encoding of cell state. Different
representations — higher-dimensional PCA, full gene expression, pathway
aggregation, or combined forms — could capture different aspects of cell-state
biology. If the per-drug r ceiling (~0.645) reflects a cell-representation
bottleneck, a better feature set should push it higher. This sweep tests
whether the ceiling is sensitive to representation choice, or whether all
reasonable representations converge to the same predictive limit.

## Experimental design

- **Model**: Ridge, 10-fold PASO drug-blind CV, no drug features
- **Representations**:
  - Baseline: RNA PCA(550) + mutation PCA(200) = 750 dims
  - PCA-1500: RNA PCA(1500) + mutation PCA(200)
  - PCA-max: RNA PCA(max) + mutation PCA(max)
  - Full RNA: all gene expression features (no PCA)
  - Pathway KEGG: KEGG pathway-level aggregation
  - RNA + Pathway: RNA PCA(550) + pathway features concatenated

## Results

| Condition | Per-drug r | Alpha | Delta vs baseline |
|-----------|:---------:|:-----:|:-----------------:|
| baseline | 0.645 | 1.0 | +0.000 |
| pca_1500 | 0.645 | 10.0 | — |
| pca_max | 0.645 | 10.0 | — |
| full_rna | 0.645 | 100.0 | — |
| pathway_kegg | 0.645 | 1.0 | — |
| rna_plus_path | 0.645 | 1.0 | — |

## Interpretation

All representations with ≥500 dimensions converge to per-drug r ≈ 0.645.
The choice of cell representation — raw RNA, PCA-compressed, pathway-aggregated,
or combined — does not meaningfully affect within-drug cell ranking. This confirms
that the per-drug r ceiling is not a representation bottleneck.

The optimal Ridge alpha varies across representations (higher alpha for
higher-dimensional inputs, reflecting the need for stronger regularization),
but the regularized performance converges regardless.


