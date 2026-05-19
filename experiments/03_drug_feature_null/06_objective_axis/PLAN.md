# PLAN: Objective Axis

## What this experiment answers

`02_representation_ablation` fixes the training objective (MSE) and varies the representation.
A skeptic can argue: "MSE is a poor match for the within-drug ranking task — it optimizes
absolute error, not rank order. A model trained with a ranking-aware loss might extract value
from drug features that MSE ignores, making the null an objective artifact rather than a
property of drug features."

This experiment fixes the representation (Morgan FP) and varies the training objective:
MSE (baseline) vs pairwise RankNet BCE. If Δ is near zero under MSE but crosses the gate
under RankNet, the binding constraint is the objective, not the representation.

## Design

| Setting | Value |
|---------|-------|
| Model | 2-layer MLP (512-dim hidden, ReLU, dropout=0.1) |
| Cell features | RNA PCA(550) + mutation PCA(200) |
| Drug features | Morgan FP (2048-bit) — fixed across conditions |
| Splits | PASO 5-fold drug-blind CV |
| Checkpoint selection | Best val per-drug r (10% drug-blind val holdout per fold) |
| Metric | Per-drug Pearson r |

Ridge is not used here because Ridge cannot implement a ranking loss. MLP with MSE is the
Ridge-equivalent baseline; MSE Δ from MLP should match Ridge Δ ≈ +0.003 (verification check
within ±0.005 tolerance; deviation > ±0.005 is a finding requiring investigation).

**MLP input structure**: cell features = RNA PCA(550) + mutation PCA(200) = 750-dim; drug
features = Morgan FP 2048-dim. Concatenation: [cell(750) ‖ drug(2048)] = 2798-dim → Linear
→ 512-dim → ReLU → Dropout(0.1) → Linear → 512-dim → ReLU → Dropout(0.1) → Linear → 1.
For `mlp_mse_no_drug`: [cell(750)] = 750-dim → 512 → 512 → 1 (same depth, no drug input).

### Objective conditions

| Condition | Loss | Notes |
|-----------|------|-------|
| `mlp_mse_no_drug` | MSE, no drug features | MLP analog of Ridge `no_drug` |
| `mlp_mse_morgan` | MSE, Morgan FP | MLP analog of Ridge `morgan_fp`; Δ should ≈ +0.003 |
| `mlp_ranknet_morgan` | Pairwise RankNet BCE, Morgan FP | Tests whether ranking loss unlocks drug feature value |

RankNet pairs are sampled within each drug (all cell-line pairs per drug per batch). Pair
sampling is balanced across drugs per batch to prevent high-response-count drugs from
dominating.

### Decision gate

Same Δ > 0.01 gate as `02`. If `mlp_ranknet_morgan` Δ > 0.01 over `mlp_mse_no_drug`,
the objective is the binding constraint — drug features provide ranking signal that MSE
cannot recover.

## Expected results

| Condition | Per-drug r | Δ |
|-----------|-----------|---|
| `mlp_mse_no_drug` | ≈ 0.63 ± 0.02 | — |
| `mlp_mse_morgan` | ≈ 0.63 ± 0.02 | ≈ +0.003 |
| `mlp_ranknet_morgan` | ≈ 0.64 ± 0.02 | ≈ +0.011 (gate boundary) |

The RankNet result is at the gate boundary from prior work (results.tex). Confirming it
establishes "objective is a binding constraint" as a second finding, distinct from the
"representation is irrelevant" finding from `02`.

## Validation checks

- `mlp_mse_morgan` Δ within ±0.005 of Ridge `morgan_fp` Δ (confirms MLP-MSE ≈ Ridge for this
  task; deviation > ±0.005 is a finding — log it and investigate before concluding on RankNet)
- `mlp_ranknet_morgan` Δ > `mlp_mse_morgan` Δ (ranking loss extracts more signal)
- Train per-drug r > 0.2 by epoch 10 (detects dead gradients, broken loss, or data pipeline
  errors — all three conditions must satisfy this)
- RankNet pair sampling: verify ≥ 90% of drugs contribute pairs in each batch

## Random seeds

- `FOLD_SEED=42` — PASO fold assignment
- `MODEL_SEED=0` — weight initialization
- `BATCH_SEED=0` — DataLoader shuffle and pair sampling

## Memory budget (DGX Spark, 128 GB unified)

| Component | Estimate | Notes |
|-----------|---------|-------|
| MLP parameters (small) | < 0.01 GB | |
| MSE batch (256 pairs × 750 cell + 2048 drug features) | < 0.1 GB | |
| RankNet pair buffer | **streaming only** — never precompute full pair matrix | see below |
| Total peak | **< 2 GB** | safe |

**RankNet pair sampler — streaming required**: with 687 cells/drug, full pair matrix per drug
= 687×686/2 ≈ 235K pairs × 233 drugs = 55M pairs. **Do not precompute the full pair matrix.**
Instead, for each batch: sample `N_pairs_per_drug` pairs per drug uniformly at random per step.
`N_pairs_per_drug = batch_size // n_drugs_in_batch` ensures balanced drug representation.
Default: 100 pairs per drug per step (≈ 0.04% of all pairs); confirmed sufficient from prior
work. **N_pairs sensitivity check**: run N_pairs ∈ {50, 100, 200} for fold 0 only (before
full run); if per-drug r at epoch 50 differs by > 0.005 across values, report the sensitivity
and choose the value maximizing val per-drug r at epoch 50.

## Required telemetry (logged per epoch)

```
### Training dynamics (per epoch per fold)
- train_loss
- val_loss
- train_per_drug_r
- val_per_drug_r          ← checkpoint selection metric
- learning_rate
- grad_norm_pre_clip
- epoch_time_s

### System (per epoch)
- gpu_memory_peak_gb
- gpu_memory_reserved_gb

### RankNet-specific (per batch, aggregated per epoch)
- pairs_per_batch_mean
- drugs_contributing_pairs_fraction  ← must be ≥ 0.90 every epoch
- pair_label_positive_fraction       ← expected ≈ 0.50

### Final (per fold)
- best_epoch
- val_per_drug_r_at_best
- test_per_drug_r_at_best
- total_train_time_h

### Outputs to disk
- logs/fold{k}_{condition}_epoch_metrics.parquet
- logs/fold{k}_{condition}_test_predictions.parquet  — drug_id, cell_id, y_true, y_pred
- checkpoints/fold{k}_{condition}_best.pt
```

## Prerequisites

- GDSC2 omics: `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/morgan_fp.npy`
- PASO splits

## Pre-flight checks

Before running, verify:
1. `data/processed/rna.parquet` and `data/processed/mutations.parquet` shapes match `02`
2. `data/processed/morgan_fp.npy` shape: (233, 2048)
3. Fold CSV files present for folds 0–4 (PASO drug-blind)
4. GPU memory available: `nvidia-smi` shows > 4 GB free
5. N_pairs sensitivity check: run fold 0 only for 50 epochs with N_pairs ∈ {50, 100, 200};
   confirm val per-drug r at epoch 50 differs by < 0.005 across values; select N_pairs=100
   if all within tolerance (else choose highest val r value)
6. 1-batch forward pass: both MSE and RankNet losses produce finite values and backward pass
   completes without error

## How to run

```bash
sbatch experiments/03_drug_feature_null/06_objective_axis/jobs/sbatch.sh
```

Runtime: ≈ 3 h (GPU, 3 conditions × 5 folds, 200 epochs each).

## Output

```
report/data/metrics.json   — per-condition per-drug r (mean, std, fold values, Δ)
logs/                      — telemetry parquets (see above)
```
