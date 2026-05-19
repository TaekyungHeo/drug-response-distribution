# PLAN: Cross-Dataset Transfer (GDSC2 → PRISM)

## What this experiment answers

All prior experiments (01–06) test drug feature null within-dataset: the model is trained
and evaluated on the same dataset (GDSC2) with drug-blind cross-validation. A skeptic could
argue: "Drug structural features might still encode generalizable pharmacological signal that
a small GDSC2 training set can't exploit. Given training data from a different dataset with
more diversity, features might transfer."

This experiment provides the strongest test: **train Ridge on all GDSC2 data, evaluate on
held-out PRISM drugs** (different dataset, different assay, largely non-overlapping drug set).
If morgan_fp ≈ no_drug here, structural features don't transfer even across dataset boundaries.
If morgan_fp > no_drug here but not within-dataset, it would suggest the structural feature
signal exists but requires cross-dataset training diversity to emerge (a nuanced, publishable finding).

## Design

| Parameter | Value |
|-----------|-------|
| Train dataset | GDSC2 (all pairs, n≈160K) |
| Test dataset | PRISM Repurposing (all pairs, n≈300K) |
| Conditions | `morgan_fp`, `no_drug` |
| Model | Ridge(α=1.0) |
| Cell features | RNA PCA(550) + mutations PCA(200), **fit on GDSC2 training cells** |
| Drug features (train) | Morgan FP for GDSC2 drugs |
| Drug features (test) | Morgan FP for PRISM drugs (computed independently) |
| Cell intersection | GDSC2 ∩ PRISM ∩ RNA ∩ mutations (expected ≈ 300–500 cells) |
| Metric | Per-drug Pearson r on PRISM test drugs |
| No cross-validation | Single GDSC2-train / PRISM-test split |

## Why this is the right comparison

- **Different assay platform**: GDSC2 (IC₅₀) → PRISM (viability AUC)
- **Largely non-overlapping drugs**: Most PRISM drugs are not in GDSC2
- **Shared cell features**: CCLE RNA/mutations available for both (same underlying biology)
- **No leakage**: PRISM test drugs were never seen during GDSC2 training

## Expected result

`morgan_fp ≈ no_drug` (Δ ≤ 0.01). If structural features don't help within-dataset (02), they
should not help cross-dataset where the training signal is even weaker. Finding Δ > 0.03 here
would be a strong counter-argument that deserves separate investigation.

## Validation checks

1. n_train_cells ≥ 400 (GDSC2 cells with RNA + mutations)
2. n_test_cells ≥ 300 (PRISM cells in GDSC2 cell intersection)
3. n_test_drugs ≥ 400 (PRISM drugs after MIN_CELLS filter)
4. No PRISM test drug appears in GDSC2 training drugs (logged; overlap ≤ 10% expected)
5. Per-drug r sanity: `no_drug` on PRISM ≈ 0.3–0.4 (lower than within-GDSC2 ≈ 0.63)
6. Ridge degenerate check: PCA fit produces non-NaN features

## Prerequisites

- `data/processed/drug_response.parquet` — GDSC2 drug response
- `data/processed/prism_drug_response.parquet` — PRISM preprocessed response
- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- Morgan FPs computed for GDSC2 and PRISM drugs via `src/data/drug_features.py`

## Output

```
report/data/metrics.json   — per-drug r on PRISM for morgan_fp and no_drug,
                             n_train, n_test, drug overlap stats
logs/run.log               — full training log
```

## How to run

```bash
sbatch experiments/03_drug_feature_null/07_cross_dataset_transfer/jobs/sbatch.sh
```

Expected runtime: < 15 min (pure Ridge, no GPU needed).
SLURM: `--mem=32G`, no GPU, `--time=0:30:00`.
