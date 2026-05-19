# PLAN: Global r vs Per-Drug r — Is Global Pearson r the Right Metric?

## Motivation

Before measuring any model's performance, we need to confirm that the metric measures what
we actually care about. In drug response prediction, the clinically relevant question is:
**"For this specific drug, which cell lines are sensitive?"** — a within-drug ranking problem.

Global Pearson r, computed across all (cell, drug) pairs, conflates two independent signals:
1. **Between-drug signal**: Which drugs have higher absolute IC50? (drug potency ranking)
2. **Within-drug signal**: For a fixed drug, which cells are sensitive? (the target task)

GDSC2 IC50 values span −5.2 to 10.3 LN µM across drugs. If between-drug variance dominates
total variance, global r is primarily rewarding the model for learning drug potency ordering —
not cell sensitivity ranking. A model could achieve high global r with zero within-drug
discriminative ability.

This experiment answers: **Is global r the right metric, or does per-drug r better capture
the target task?**

---

## Design Overview

Four independent lines of evidence, each blocking a different class of counterarguments:

| Step | Evidence | Blocks |
|------|----------|--------|
| 1 | Variance decomposition (no model) | "global r measures both signals equally" |
| 2 | Baseline dissociation (no model) | "the two metrics measure the same thing" |
| 3 | Multi-model consistency | "this is an artifact of model underfitting/capacity" |
| 4 | Cross-split consistency | "this is a drug-blind-specific artifact" |

Statistical rigor (bootstrap CI + paired fold test) applies to Steps 3 and 4.

---

## Step 1 — Variance Decomposition (data only, no model)

**Goal**: Show mathematically that between-drug variance dominates GDSC2 IC50,
making global r a poor proxy for within-drug ranking.

**Method**: Apply the law of total variance to IC50:

```
Var(y) = E_d[Var_c(y|d)]   +   Var_d[E_c(y|d)]
       = within-drug var   +   between-drug var
```

Compute both components from the training data. Then derive the ceiling on global r
attributable to between-drug signal alone:

```
global_r_ceiling_from_between_drug ≈ sqrt(between_drug_var / total_var)
```

If this ceiling is ≥ the model's actual global r, the model is not doing better than
a drug-mean oracle on global r — i.e., global r is measuring drug potency, not cell ranking.

**Output**: `results/variance_decomposition.json` with:
- `between_drug_var_fraction`: between-drug variance / total variance
- `within_drug_var_fraction`: within-drug variance / total variance
- `global_r_ceiling_from_between`: sqrt(between_drug_var_fraction)
- Per-drug IC50 mean and std

**Job**: `jobs/compute_variance_decomposition.py`

**Expected finding**: between-drug variance fraction ≈ 0.5–0.7, meaning global r is
substantially driven by between-drug signal.

---

## Step 2 — Baseline Dissociation (no model training required)

**Goal**: Construct two extreme predictors that demonstrate global r and per-drug r
measure independent signals. This is data/arithmetic — no GPU required.

### Predictor 2A: per-drug-mean predictor (mixed_set or cell_blind)

```
ŷ(c, d) = mean IC50 of drug d over training cells
```

Run on **mixed_set** (or cell_blind). Cannot run on drug_blind because test drugs
have no training mean.

This predictor has zero within-drug discriminative ability: it predicts the same value
for all cells given a drug.

- **Expected global r**: high (correctly ranks drugs by potency)
- **Expected per-drug r**: undefined (constant prediction → zero std → Pearson undefined)
  → **Policy**: treat undefined per-drug r as 0.0 (pre-registered before running)

**Validation against Step 1**: the empirical global r of Predictor 2A should match
`sqrt(between_drug_var / total_var)` from Step 1 within ±0.02. If not, the decomposition
formula does not hold for this dataset and Step 1 results should be re-examined.

### Predictor 2B: drug-mean-removed predictor (any split; use existing drug_blind results)

Take the already-trained OmniCancerV1 predictions from `report/data/results.json`
(drug_blind, standard condition, folds 0–4). No retraining needed.

Subtract the per-drug mean of the **test-set predictions** for each drug d:

```
ŷ_resid(c, d) = ŷ(c, d) − (1/|C_test_d|) ∑_{c' ∈ C_test_d} ŷ(c', d)
```

where C_test_d is the set of test cells for drug d. This removes the between-drug
signal from predictions while leaving within-drug ranking intact.

- **Expected global r**: low (between-drug signal removed from predictions)
- **Expected per-drug r**: preserved (subtracting a per-drug constant does not change
  within-drug ranking, so Pearson r within each drug is unchanged)

**Interpretation**: Predictor 2A achieves high global r / ~0 per-drug r; Predictor 2B
achieves low global r / high per-drug r. Same model, same data — only what signal is
retained differs. This proves the two metrics measure independent phenomena.

**Output**: `results/baseline_dissociation.json`

**Job**: `jobs/compute_baseline_dissociation.py` (fast, ~2 min, no GPU)

---

## Step 3 — Multi-Model Consistency (blocks capacity/underfitting argument)

**Goal**: Show the global r vs per-drug r gap is consistent across model classes
of varying capacity. If the gap were due to model underfitting, larger models would
close it. If the gap is metric-structural, it persists regardless of capacity.

**Models** (all use **identical features**: Morgan FP + RNA + mutations):
- Ridge regression (linear baseline)
- MLP-Small: `[RNA+mut dim → 128 → 64 → 1]`, dropout=0.1
- MLP-Medium: `[RNA+mut dim → 512 → 256 → 64 → 1]`, dropout=0.1
- MLP-Large: `[RNA+mut dim → 2048 → 512 → 128 → 1]`, dropout=0.1
- OmniCancerV1: attention-based, same features (existing trained model, fold 0–4)

**Split**: drug-blind, 5 folds, same PASO fold files for all models.

**Training** (MLP models):
- MAX_EPOCHS=200, early stop on val per-drug r (patience=20)
- LR=1e-3 cosine → 1e-5, batch_size=512, weight_decay=1e-4
- Report best val epoch per fold

**Metrics per model**: global_r (mean ± std across folds), per-drug r (mean ± std), gap

**Output**: `results/model_comparison.json`

**Job**: `jobs/run_model_comparison.py`

**Expected finding**: gap (per-drug r − global r) ≈ +0.14 across all model sizes.

---

## Step 4 — Cross-Split Consistency (blocks "drug-blind artifact" argument)

**Goal**: Show the gap appears on all three evaluation splits, not just drug-blind.

**Splits**:
- `mixed_set`: random 80/20 split, both drugs and cells overlap train/test
- `cell_blind`: held-out cell lines (test cells never seen during training)
- `drug_blind`: held-out drugs (test drugs never seen during training)

**Model**: OmniCancerV1, standard condition (raw LN IC50 targets).

**Folds**: 5 folds per split.

**Output**: `results/cross_split_consistency.json`

**Job**: `jobs/run_cross_split.py` (extends existing `jobs/run.py`)

**Expected finding**: gap is positive and consistent across all splits.

---

## Step 5 — Statistical Rigor

Applied to Steps 3 and 4 results.

**Bootstrap CI** (B=1000):
- Bootstrap over folds to get 95% CI on mean gap
- Report `gap_mean`, `gap_ci_lower`, `gap_ci_upper` per condition

**Paired fold-level test**:
- For each fold i: `gap_i = per_drug_r_i − global_r_i`
- One-sample t-test: H0: mean(gap_i) = 0
- Report t-statistic and p-value

**Drug count reporting**:
- `n_drugs_evaluated`: number of drugs with ≥5 test samples (included in per-drug r)
- `n_drugs_excluded`: number of drugs excluded (< 5 samples)
- Per-fold and mean

**Job**: integrated into `run_model_comparison.py` and `run_cross_split.py`

---

## Pre-registered Decision Criteria

Before running any job, the following thresholds are committed:

1. **Variance decomposition**: if `between_drug_var_fraction > 0.50`, conclude global r
   is structurally dominated by between-drug signal.
2. **Baseline dissociation**: per-drug-mean predictor per-drug r = 0.0 by policy
   (undefined → 0 imputation).
3. **Multi-model gap**: if gap > 0.10 for all 5 model classes, metric structure
   is the cause (not capacity/training).
4. **Cross-split gap**: if gap > 0.05 on all 3 splits, not a split-specific artifact.
5. **Primary metric**: if criteria 1–4 are all satisfied, per-drug r replaces global r
   as the primary metric for all subsequent experiments.

---

## Supplemental: Motivation for per-drug r as Clinical Metric

Per-drug r maps directly to the clinical use case: given a specific drug under consideration
for a patient, rank candidate patients by predicted sensitivity. Between-drug ranking
(which drug is generally more potent) is captured by global r but is irrelevant to this task —
drug selection is a separate decision made on pharmacological and safety grounds, not
predicted IC50 alone.

---

## File Structure

```
01_global_vs_perdrug/
├── PLAN.md                          (this file)
├── jobs/
│   ├── compute_variance_decomposition.py   # Step 1
│   ├── compute_baseline_dissociation.py    # Step 2
│   ├── run_model_comparison.py             # Step 3 + Step 5
│   ├── run_cross_split.py                  # Step 4 + Step 5
│   └── run.py                             # Original Step 3/4 combined (OmniCancerV1 only)
├── results/
│   ├── variance_decomposition.json
│   ├── baseline_dissociation.json
│   ├── model_comparison.json
│   ├── cross_split_consistency.json
│   └── run_<timestamp>/results.json       # Per-run outputs
├── logs/
└── report/
    ├── README.md
    └── data/results.json
```

---

## How to Run

```bash
# Step 1: variance decomposition (fast, ~1 min)
uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/jobs/compute_variance_decomposition.py

# Step 2: baseline dissociation (fast, ~2 min)
uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/jobs/compute_baseline_dissociation.py

# Step 3: multi-model comparison (slow, ~4–6 hours on NVIDIA GB10)
uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/jobs/run_model_comparison.py

# Step 4: cross-split consistency (slow, ~3 hours on NVIDIA GB10)
uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/jobs/run_cross_split.py

# Original OmniCancerV1 run (already completed — results in report/)
uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/jobs/run.py --condition both
```

Run Steps 1 and 2 first (fast, no GPU needed). Steps 3 and 4 can run in parallel on separate
GPU sessions.

---

## Dependencies

- `data/processed/rna.parquet`
- `data/processed/mutations.parquet`
- `data/processed/gdsc2_response.parquet`
- `data/processed/cell_line_index.parquet`
- PASO drug-blind fold files: `external/PASO/data/10_fold_data/drug_blind/DrugBlind_{train,test}_Fold{0..4}.csv`
- Mixed-set and cell-blind fold files: `external/PASO/data/10_fold_data/`
- `src/data/omics_utils.py`, `src/data/splits.py`
- `src/evaluation/per_drug.py`, `src/evaluation/metrics.py`
- `src/models/omnicancer.py`

All processed data must be generated first via:
```bash
uv run python3 experiments/00_data_preparation/jobs/download.py
uv run python3 experiments/00_data_preparation/jobs/preprocess.py
```

---

## Known Completed Work

The original `jobs/run.py` has already run Steps 3+4 for OmniCancerV1 (drug-blind split,
5 folds, standard + per_drug_zscore conditions). Results are in `report/data/results.json`.
Steps 1, 2, and the full multi-model/cross-split expansions are new.
