# 03 — Baselines

Establishes the cell-only prediction ceiling under per-drug Pearson r across three
evaluation protocols: mixed-set, cell-blind, and drug-blind.

## Purpose

After selecting per-drug r as the primary metric (02_metric_selection), this experiment
asks: what does a model that uses only cell features (no drug features) achieve? And how
does that compare to baselines?

## Results

### Baselines

| Baseline | Split | Per-drug r | Global r |
|----------|-------|:----------:|:--------:|
| Cell-mean prior | drug-blind | 0.652 | 0.268 |
| Drug-mean oracle | mixed-set | — | 0.838 |

Cell-mean prior (drug-blind per-drug r = 0.652): predicts each cell's average IC50
across training drugs, ignoring drug identity entirely. This is the correct null: any
model that correctly learns cell sensitivity will do at least this well.

Drug-mean oracle (global r = 0.838): predicts each test drug's training mean, ignoring
cell identity. Illustrates how much of global r is trivially captured by drug potency.

### Ridge regression

| Split | Per-drug r | Global r |
|-------|:----------:|:--------:|
| Mixed-set | 0.644 | 0.322 |
| Cell-blind | 0.463 | 0.229 |
| Drug-blind | 0.645 | 0.344 |

Ridge (RNA-seq, 19,193 features) matches the cell-mean prior per-drug r in the
drug-blind setting (0.645 vs 0.652 oracle). This confirms Ridge is learning mean cell
sensitivity rather than drug-specific cell ranking — it is operating as a sophisticated
cell-mean estimator.

Note: The paper's canonical cell-blind Ridge per-drug r = 0.438 (reported as `\CellBlindR`)
comes from the dedicated 5-fold cell-blind CV in
`04_cell_representation/01_ceiling_characterization/01_split_ceilings`, which uses
proper cell-blind splits. The 0.463 above is from this experiment's mixed-design 5-fold CV.

## Key finding

The cell-mean prior holdout oracle achieves per-drug r = 0.652 (using test-set drug means,
not practical in deployment). Ridge drug-blind achieves r = 0.645 (drug-blind 10-fold CV
using only training-set means), which is the practical ceiling and the K=0 baseline used
throughout the paper. The 0.007 gap between the oracle and Ridge confirms Ridge is learning
mean cell sensitivity rather than drug-specific cell rankings. Any improvement in per-drug r
above 0.645 requires drug-specific information that shifts cell rankings beyond the mean
sensitivity profile.

## Input data

- `data/processed/rna.parquet`, `mutations.parquet`, `drug_response.parquet`
- `external/PASO/data/10_fold_data/drug_blind/` — PASO drug-blind splits
- `external/PASO/data/10_fold_data/cell_blind/` — cell-blind splits

## Output files

- `report/data/metrics.json` — oracle and Ridge results under all three split types
