# PLAN: Decompose PASO's r = 0.745

## What this experiment answers

PASO (Wu et al., PLoS CB 2025) reports a drug-blind Pearson r = 0.745 on GDSC IC₅₀.
This experiment measures how much of that figure is attributable to two protocol
artifacts rather than model quality:

1. **Test-set snooping**: PASO selects its final checkpoint by monitoring test-set
   Pearson at every epoch. No held-out validation set is used. The reported r is
   the best-ever test r across all epochs, not a generalization estimate.

2. **Best-fold cherry-picking**: PASO reports only the single best fold (fold 8,
   r ≈ 0.745) rather than the 10-fold mean.

The experiment isolates each artifact by running OmniCancerV1 under two protocols on
PASO's own fixed drug-blind 10-fold splits:

| Protocol | Checkpoint selection | Evaluates |
|----------|----------------------|-----------|
| PASO-style | Best test Pearson each epoch | snooping inflation |
| Fair | Best val Pearson (drug-blind val) | unbiased estimate |

Snooping Δ = PASO-style mean − fair mean. Best-fold Δ = best single fold − fair mean.

## Why a drug-blind val split

The fair protocol holds out **10% of training drugs entirely** as validation, not 10%
of random pairs. This design choice is essential to the measurement:

- If val is a random pair split, val drugs overlap with training drugs. Val r lands
  near 0.91 (train-drug distribution) while test r stays near 0.34 (drug-blind
  distribution). The two are on different scales — val is not a useful signal for
  when to stop, and the inflation comparison is contaminated.

- With drug-blind val, val and test are drawn from the same distribution (held-out
  drugs). Val r tracks test r. Checkpoint selection via val r is a fair proxy for
  test r, and the snooping Δ reflects only the test-monitoring artifact.

## Global r vs per-drug r: role separation

**Checkpoint selection uses global r throughout** — both PASO-style (test global r)
and fair (val global r). This is intentional: we are reproducing PASO's *protocol*,
and PASO used global r for selection. Changing the selection metric would conflate
two different design axes.

**Per-drug r is computed as a secondary post-hoc metric** at the selected checkpoint.
It answers: "what is the within-drug cell-ranking ability of the chosen model?"
Primary result tables report global r for apples-to-apples comparison with PASO's
headline. Per-drug r is reported alongside to show metric inflation independently.

## Model and hyperparameters

OmniCancerV1 (transformer encoder, Morgan FP drug representation) with RNA +
mutations. Hyperparameters are our standard choice, not tuned to PASO's data — the
goal is to measure the *protocol* artifact, not to compete on absolute r.

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 512 |
| LR | 1e-3 |
| d_model | 256 |
| n_heads | 8 |
| n_layers | 4 |
| Dropout | 0.1 |
| Modality dropout | 0.3 |
| Drug features | Morgan FP (2048-bit radius-2) |
| Omics | RNA + mutations |
| Folds | 10 (PASO's exact splits) |
| Val fraction | 10% of training drugs (drug-blind; seed = 42 + fold_i) |

## Note on `drop_last`

`02_decomposition` uses a custom `eval_set()` that iterates all indices without
dropping any pairs. This is intentional — we want full-set evaluation.

`01_reproduction` (PASO faithful) uses `DataLoader(..., drop_last=True)` for the
test loader, which matches PASO's original `train_PASO_Kfold_double_omics.py` line 130.
The two scripts differ on this deliberately.

## Prerequisites

### 1. PASO submodule (fold splits only — no PASO code needed here)

```bash
cd ~/multi-onco
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init external/PASO
```

Required files:
```
external/PASO/data/10_fold_data/drug_blind/DrugBlind_train_Fold{0-9}.csv
external/PASO/data/10_fold_data/drug_blind/DrugBlind_test_Fold{0-9}.csv
```

### 2. Processed omics (from `experiments/00_data_preparation`)

```
data/processed/rna.parquet
data/processed/mutations.parquet
data/processed/cell_line_index.parquet
data/processed/morgan_fp.npy
```

If missing, run:
```bash
sbatch experiments/00_data_preparation/jobs/sbatch.sh
```

### 3. Node setup (spark1 / spark2 do not share disks)

```bash
git clone git@github.com:TaekyungHeo/multi-onco.git ~/multi-onco
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/multi-onco
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init external/PASO
# then copy or re-run 00_data_preparation to populate data/processed/
```

## How to run

```bash
cd ~/multi-onco
sbatch experiments/02_reproductions/01_paso/02_decomposition/jobs/sbatch.sh
```

Runtime: ~6–10 h on a single GPU (2 protocols × 10 folds × 200 epochs).

## Output

```
results/run_<timestamp>/results.json   — raw per-fold results
report/data/results.json               — copy of latest run (for metrics.py)
report/data/metrics.json               — aggregated summary (run metrics.py)
report/data/metadata.json              — run metadata (git hash, config)
```

`metrics.json` fields:
```json
{
  "n_folds": 10,
  "paso_reported_r": 0.745,
  "paso_style_mean": ...,    // global r, best test checkpoint (snooping)
  "paso_style_std": ...,
  "fair_mean": ...,           // global r, best val checkpoint (fair)
  "fair_std": ...,
  "inflation": ...,           // paso_style_mean - fair_mean
  "best_fold_test_r": ...,   // highest single-fold r under PASO-style
  "best_fold_val_r": ...,
  "paso_style_per_drug_mean": ...,  // mean per-drug r at snooping checkpoint
  "fair_per_drug_mean": ...          // mean per-drug r at fair checkpoint
}
```

To regenerate `metrics.json` from the latest run:
```bash
cd ~/multi-onco
uv run python experiments/02_reproductions/01_paso/02_decomposition/metrics.py
```

## Expected results

| Metric | Value |
|--------|-------|
| PASO-style global r mean | ≈ 0.560 |
| Fair global r mean | ≈ 0.521 |
| Snooping Δ (global r) | ≈ +0.039 |
| Best single fold (PASO-style) | ≈ 0.745 |
| Best-fold Δ (fair mean → best fold) | ≈ +0.189 |
| Total inflation (fair mean → headline) | ≈ +0.228 |

Note: per-drug r expectations will be populated after the first full run.

## Validation checks

- [ ] PASO-style mean > fair mean (snooping inflation is positive)
- [ ] Best single fold r ≈ 0.745 (reproduces PASO's headline within ±0.01)
- [ ] Fair per-drug r < fair global r by ≥ 0.10 (global-r metric inflation present)
- [ ] Val drug set ∩ train drug set = ∅ for every fold (verified by unit tests)
