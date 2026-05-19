# 03 — RPPA proteomics oracle: protein-level data vs transcriptomics

## Research question

Does protein-level data (RPPA reverse-phase protein arrays, 214 proteins)
improve per-drug r beyond RNA transcriptomics? If the per-drug r ceiling
reflects post-transcriptional regulation invisible to RNA-seq, protein
measurements should break it.

## Background

Both the representation sweep (01) and the foundation model test (02) show
that all RNA-based representations converge to the same per-drug r ceiling.
A fundamental objection remains: drug sensitivity is ultimately determined
by protein expression and activity — post-transcriptional regulation that
RNA-seq cannot capture. If the ceiling is a modality bottleneck rather than
a measurement noise or training distribution issue, protein-level data should
provide independent information and push per-drug r higher. This is the
strongest test of the representation ceiling hypothesis using biological
ground truth.

## Experimental design

- **Model**: Ridge, 10-fold PASO drug-blind CV
- **Conditions**:
  - A: RNA PCA(550) + mutation PCA(200) (baseline)
  - B: RPPA only (214 proteins)
  - C: RNA + mutation + RPPA (concatenated)

## Results

| Condition | Per-drug r | Delta vs A |
|-----------|:---------:|:----------:|
| A: RNA + mutation PCA | 0.647 | — |
| B: RPPA only (214 proteins) | 0.557 | — |
| C: RNA + mutation + RPPA | 0.647 | — |

## Interpretation

RPPA alone achieves 0.557 — lower than RNA (0.647),
reflecting the limited panel size (214 proteins vs ~19,000 genes).
Concatenating RPPA with RNA+mutation yields no improvement.

The per-drug r ceiling is robust across representation modalities: neither
transcriptomics nor proteomics nor their combination can break it. The
Apoptosis regulation ceiling (profile concordance r = 0.33) reflects genuine
biological heterogeneity, not a representation artifact.


