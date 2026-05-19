# 01 — Nonlinear Models: Is the Ceiling Ridge-Specific?

## Research question

Is the drug-blind per-drug r ceiling (~0.645) an artifact of Ridge's linearity,
or a structural property of the drug-blind problem?

## Background

The per-drug r ceiling (~0.645) was established using Ridge regression, a
linear model. Ridge can capture only additive cell-feature effects. Nonlinear
models (XGBoost, MLP) can represent interaction effects — for example, specific
combinations of gene expression patterns that jointly predict sensitivity to a
drug class. If the ceiling is a linearity artifact, nonlinear models should
exceed it substantially. This is a critical robustness check: the paper's
claim that the ceiling is near the measurement noise limit (83.7% of replicate
concordance) requires that more expressive models do not close the gap.

## Experimental design

- **Model**: Ridge(alpha=1.0), XGBoost, MLP (2 hidden layers, 512 units, ReLU)
- **Cell features**: RNA PCA(550) + mutation PCA(200)
- **Data**: GDSC2, 10-fold PASO drug-blind CV
- **Metric**: per-drug Pearson r

## Results

| Condition | Per-drug r | delta vs Ridge |
|-----------|:---:|:---:|
| Ridge | 0.645 ± 0.025 | — |
| XGBoost | 0.645 ± 0.025 | -0.000 |
| MLP | 0.639 ± 0.025 | -0.006 |

**The ceiling is not Ridge-specific.** XGBoost and MLP produce the same per-drug r
as Ridge (delta < 0.01). Nonlinearity does not break the ceiling.

### Fold-level values

| Fold | Ridge | XGBoost | MLP |
|:---:|:---:|:---:|:---:|
| 0 | 0.601 | 0.600 | 0.595 |
| 1 | 0.613 | 0.613 | 0.606 |
| 2 | 0.651 | 0.651 | 0.648 |
| 3 | 0.637 | 0.637 | 0.627 |
| 4 | 0.661 | 0.661 | 0.653 |
| 5 | 0.649 | 0.649 | 0.644 |
| 6 | 0.624 | 0.624 | 0.620 |
| 7 | 0.670 | 0.670 | 0.667 |
| 8 | 0.672 | 0.672 | 0.663 |
| 9 | 0.675 | 0.675 | 0.670 |

## Interpretation

All three model classes converge to per-drug r ≈ 0.645. XGBoost can capture
feature interactions, and MLP can represent arbitrary nonlinear functions of
cell features — yet neither improves on Ridge. This rules out linearity as an
explanation for the ceiling. Combined with the measurement noise ceiling
(replicate concordance r = 0.754, Ridge at 83.7%), this confirms that the
remaining gap is noise-limited, not capacity-limited.


