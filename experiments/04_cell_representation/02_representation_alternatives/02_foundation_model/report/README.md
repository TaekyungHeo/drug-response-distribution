# 02 — scFoundation: foundation model embeddings vs RNA PCA

## Research question

Does replacing RNA PCA features with scFoundation single-cell foundation model
embeddings (768-dim, pretrained on 50M cells) improve drug-blind per-drug r?

## Background

The representation sweep (01) showed that all ≥500-dim representations
converge to the same per-drug r. A skeptic can argue these are all simple
linear transformations of the same RNA-seq data. Large foundation models
pre-trained on tens of millions of single-cell transcriptomes (scFoundation:
50M cells, 768-dim embeddings) may capture non-linear functional cell-state
structure that PCA misses. If the per-drug r ceiling is a representation
bottleneck rather than a measurement noise or training distribution issue,
foundation model embeddings should break it.

## Experimental design

- **Model**: Ridge, 10-fold PASO drug-blind CV
- **Conditions**:
  - A: RNA PCA(550) + mutation PCA(200) (baseline)
  - B: scFoundation embeddings (768-dim)
  - C: RNA PCA(550) + mutation PCA(200) + scFoundation (concat)

## Results

| Condition | Per-drug r | Delta vs A |
|-----------|:---------:|:----------:|
| A: RNA + mutation PCA | 0.650 | — |
| B: scFoundation (768-dim) | 0.650 | +0.000 |
| C: RNA + mut + scFoundation | 0.650 | +0.000 |

## Interpretation

scFoundation embeddings achieve the same per-drug r as standard RNA PCA
(delta ≈ +0.000). Concatenating foundation embeddings with
RNA+mutation features provides no additional benefit. The per-drug r ceiling
is not a cell-representation bottleneck — even a foundation model pretrained
on 50 million single-cell transcriptomes cannot break it.


