# Data Sufficiency

Is the drug-blind per-drug r ceiling (r=0.645) data-limited or information-theoretic?

All experiments use Ridge(α=1.0), PASO 10-fold drug-blind CV, per-drug Pearson r.

| # | Experiment | Question | Key result |
|---|-----------|---------|------------|
| 01 | `01_learning_curve/` | Does r rise with more training cells? | r rises monotonically (0.41→0.645) with no clear plateau before fraction=1.0 |

Learning curve results (fraction of training cells used):

| Fraction | Per-drug r |
|:--------:|:----------:|
| 0.10 | 0.415 |
| 0.20 | 0.463 |
| 0.40 | 0.528 |
| 0.60 | 0.576 |
| 0.80 | 0.613 |
| 1.00 | 0.645 |

The rate of improvement is decreasing (diminishing returns), but no clear plateau is reached
before using all available cells. The ceiling has a data-limited component — additional
cell-line coverage could improve performance further.
