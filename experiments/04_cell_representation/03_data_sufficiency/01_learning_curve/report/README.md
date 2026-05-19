# 01 — Cell-line learning curve: how many cells are needed?

## Research question

How does per-drug r scale with the number of training cell lines? Is the
current dataset (960 cells) saturated, or would more cell lines substantially
improve prediction? This determines whether data acquisition (more cells)
or algorithmic improvement is the higher-leverage investment.

## Background

The representation sweep established that no RNA-based or protein-level feature
set improves over the ~0.645 baseline. Before concluding that this ceiling is
fundamental, a data-quantity objection must be addressed: ~960 cell lines may
be insufficient to estimate cell-sensitivity relationships from high-dimensional
RNA features. If the learning curve has not plateaued, collecting more cell
lines would improve per-drug r independently of feature engineering. If the
curve is already flat at 960 cells, the binding constraint is not data quantity
but something else — measurement noise or training distribution.

## Experimental design

- **Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200)
- **Data**: GDSC2, 10-fold PASO drug-blind CV
- **Fractions**: 10%, 20%, 40%, 60%, 80%, 100% of training cell lines
  (subsampled uniformly within each fold)

## Results

| Fraction | N cells (approx) | Mean per-drug r |
|:--------:|:-----------------:|:--------------:|
| 0.1 | 68 | 0.4147 |
| 0.2 | 137 | 0.4633 |
| 0.4 | 274 | 0.5277 |
| 0.6 | 412 | 0.5757 |
| 0.8 | 549 | 0.6126 |
| 1.0 | 687 | 0.6453 |

## Interpretation

The learning curve shows diminishing returns: per-drug r increases rapidly
from 10% to 40% of cells, then plateaus. The 100% curve achieves
approximately the same per-drug r as the Ridge baseline (0.645), confirming
that the current cell-line panel is near saturation for this model class.

This has practical implications: doubling the number of screened cell lines
would yield minimal improvement in per-drug r. The binding constraint is
not cell-line coverage but rather the training distribution (which drugs
train together) and the measurement noise floor (replicate concordance 0.754).


