# PLAN: Learning Curve — Is the Drug-Blind Ceiling Data-Limited?

## Hypothesis

The drug-blind per-drug r ceiling of 0.631 may be data-limited: more cell lines could
push r higher. Alternatively the ceiling may be information-theoretic: it reflects the
actual predictability of drug sensitivity from transcriptomics, and more data won't help.

Distinguishing these two cases matters for the conclusion: if data-limited, collecting
more omics measurements would improve predictions; if information-theoretic, the ceiling
is biological, not technical.

## Design

- **Model**: Ridge(α=1.0), RNA PCA(550) + mut PCA(200)
- **Split**: PASO 10-fold drug-blind CV (233 drugs)
- **Variable**: training cell fraction — 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 of available cells
- **Metric**: per-drug Pearson r (averaged over test drugs, then over folds)
- **Sampling**: stratify by cell line (random sample without replacement, seed=42)

Within each fold, the training drug-response pairs are downsampled by randomly selecting
a fraction of the training cell lines. Test set and drug split are unchanged.

If the learning curve plateaus before fraction=1.0, the ceiling is information-theoretic.
If per-drug r is still rising at fraction=1.0, it is data-limited.

## How to run

```bash
~/.local/bin/uv run python3 experiments/04_cell_representation/03_data_sufficiency/01_learning_curve/jobs/run.py
```

Expected runtime: ~30 min (6 fractions × 10 folds, Ridge CPU)

## Validation checks

- fraction=1.0 per-drug r ≈ 0.631 (matches canonical baseline)
- fraction=0.1 per-drug r substantially lower (confirms sensitivity to cell count)
- Plateau pattern (or lack thereof) directly answers the data-sufficiency question

## Output

`report/data/results.json` — per fraction: mean/std per-drug r over 10 folds, n_train_cells mean

## Dependencies

- `data/processed/` omics parquets (RNA, mutations)
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/utils/ridge.compress_cell`, `src/utils/paso_folds.load_paso_pairs`

## Resources

- `--cpus-per-task=8`
- `--mem=48G` (6 fractions × 10 folds; conservative for subsampling overhead)
- `--time=4:00:00`
- No GPU needed

## Context in Paper

§Discussion: If plateau → biological ceiling, motivates active learning over data collection.
If still rising → data collection is the next lever.
