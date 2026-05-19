# PLAN: Model Robustness

## What this experiment answers

`02_representation_ablation` uses Ridge regression throughout. A skeptic can argue: "Ridge is
linear — it cannot learn complex interactions between drug features and cell features.
A nonlinear model might extract value from drug representations that Ridge ignores."

This experiment replaces Ridge with OmniCancerV1 (Transformer encoder) while keeping the
comparison simple: Morgan FP vs no_drug. If Δ is still near zero under a nonlinear model with
far higher capacity, the null is not a Ridge artifact.

The experiment also serves a secondary purpose: it produces a drug-blind-trained OmniCancerV2
checkpoint whose GNN drug embeddings are used as the `gnn` condition in `02_representation_ablation`.

## Design

### Part A — End-to-end Transformer (primary falsification)

| Setting | Value |
|---------|-------|
| Model | OmniCancerV1 (Transformer encoder, 4 layers, 256-dim, 8 heads) |
| Conditions | `morgan_fp` (2048-bit) vs `no_drug` (zero vector) |
| Cell features | RNA + mutations (all features, no PCA — Transformer handles dim) |
| Splits | PASO 5-fold drug-blind CV |
| Epochs | 200 |
| Checkpoint selection | Best val per-drug r (10% drug-blind val holdout per fold) |
| Metric | Per-drug Pearson r |

**Input projection architecture**:
- RNA (≈19K genes): `Linear(n_genes → 256)` = cell_rna_embed
- Mutations (≈700 features): `Linear(n_mut → 256)` = cell_mut_embed
- Drug (2048-bit Morgan FP for `morgan_fp`; zero vector for `no_drug`): `Linear(2048 → 256)` = drug_embed
- Input tokens to Transformer: [cell_rna_embed, cell_mut_embed, drug_embed] — 3 tokens × 256-dim
- 4-layer Transformer encoder (self-attention across the 3 tokens)
- Prediction head: mean-pool token outputs → `Linear(256 → 1)`
- Total parameters ≈ 21M (dominated by the RNA projection layer: 19K × 256 ≈ 4.9M params)

**Decision gate**: same as `02`: Δ > 0.01.

### Part B — OmniCancerV2 GNN checkpoint extraction (dependency for `02`)

After Part A, train OmniCancerV2 (replaces Morgan FP linear encoder with 3-layer GCN):
- **5-fold drug-blind training** — one checkpoint per fold, using only that fold's train drugs
- For each fold k: extract 256-dim embeddings for the fold-k test drugs from the fold-k checkpoint
- Concatenate across folds → all 233 GDSC2 drugs covered, each embedded from a checkpoint that
  never saw that drug in training
- Save to `data/processed/gnn_embeddings_256.npy`

These embeddings are then used as the `gnn` condition in `02_representation_ablation`.

## Why 5-fold GNN extraction (not single-fold)

A single-fold checkpoint (e.g., fold 0 train) never sees fold 0 test drugs, but fold 0 test
drugs appear as train drugs in folds 1–4. Using one checkpoint means embeddings for ~80% of
drugs come from a checkpoint that trained on those drugs — an information leak. 5-fold
extraction ensures every drug is embedded by a checkpoint that never trained on it.

## Validation checks

- Part A `morgan_fp` per-drug r Δ < 0.01 (null holds in Transformer)
- Part A `no_drug` per-drug r ≈ `02` Ridge `no_drug` ± 0.02 (both methods near same baseline)
- Part B GNN checkpoint drug-blind r ≈ 0.41 (GCN end-to-end known result)
- Part B extracted embeddings: 233 drugs × 256 dims

## Random seeds

All seeds fixed for reproducibility:
- `FOLD_SEED=42` — PASO fold assignment (inherited from PASO; do not change)
- `MODEL_SEED=0` — weight initialization
- `BATCH_SEED=0` — DataLoader shuffle

## Memory budget (DGX Spark, 128 GB unified)

| Component | Estimate | Notes |
|-----------|---------|-------|
| RNA features (no PCA, ~19K genes × 687 cells) | ≈ 0.1 GB | float32 |
| Model parameters (21M) | ≈ 0.08 GB | bf16 |
| Activations per batch (batch=256 pairs) | ≈ 0.3 GB | |
| Total peak | **< 4 GB** | safe on 128 GB unified |

- Mixed precision: **bf16** throughout
- Batch size: 256 (drug, cell) pairs; increase to 512 if memory allows
- No gradient accumulation needed at this scale

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

### Final (per fold)
- best_epoch
- val_per_drug_r_at_best
- test_per_drug_r_at_best
- model_param_count

### Outputs to disk
- logs/fold{k}_epoch_metrics.parquet  — one row per epoch
- logs/fold{k}_test_predictions.parquet  — drug_id, cell_id, y_true, y_pred
- checkpoints/fold{k}_best.pt
```

## Prerequisites

- GDSC2 omics: `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- Molecular graphs: `data/processed/drug_graphs.npz` (74-dim atom features)
- PASO splits

## Pre-flight checks

Before running, verify:
1. `data/processed/rna.parquet` shape: (≥ 600 cells, ≥ 18000 genes)
2. `data/processed/mutations.parquet` shape: (≥ 600 cells, ≥ 600 genes)
3. Fold CSV files present for folds 0–4 (PASO drug-blind)
4. Part B only: `data/processed/drug_graphs.npz` present; spot-check atom feature dim = 74
5. GPU memory available: `nvidia-smi` shows > 8 GB free before launch
6. 1-batch forward pass with fold 0 train data and batch_size=256 completes (confirms input
   projection shapes are correct before committing to full 5-fold run)

## How to run

```bash
# Part A: 5-fold Transformer ablation
sbatch experiments/03_drug_feature_null/03_model_robustness/jobs/sbatch_partA.sh

# Part B: GNN checkpoint + embedding extraction (5-fold, one checkpoint per fold)
sbatch experiments/03_drug_feature_null/03_model_robustness/jobs/sbatch_partB.sh
```

Runtime: Part A ≈ 10 h (GPU, 2 conditions × 5 folds × 200 epochs). Part B ≈ **10 h** (5-fold GCN training × ≈2 h/fold).

## Output

```
report/data/partA_metrics.json          — per-fold morgan_fp vs no_drug per-drug r
data/processed/gnn_embeddings_256.npy   — GNN drug embeddings for 02_representation_ablation
logs/                                   — telemetry parquets (see above)
```

---

### Part C — Extended representation ablation in Transformer

**What this answers**: Part A showed Morgan FP Δ ≈ +0.008 (null) in Transformer. The MoA
dissociation experiment (05_solutions/02_training_distribution/04_transformer_moa) showed MoA
one-hot Δ ≈ +0.027 overall and per-MoA gains up to +0.37. This raises the question: is the
dividing line "structural vs mechanistic" or "continuous vs categorical"?

Part C tests the remaining representation types under the same Transformer architecture:

| Condition | Type | Dim | Info content | Expected |
|-----------|------|-----|--------------|----------|
| `lincs_pca64` | Functional, continuous | 64 | What drug does to cells | Likely null |
| `drug_target` | Mechanistic, sparse binary | ~5145 | Which proteins hit | Unclear |
| `moa_onehot` | Mechanistic, categorical | 24 | Drug class label | Works (confirmed) |

**Design**: Same as Part A (OmniCancerV1, 10-fold PASO, 200 epochs, val per-drug r checkpoint).
Drug input projection layer adapted per condition (Linear(dim → 256)). LINCS condition uses
only the 104 matched drugs (restricted test set, same as Ridge LINCS experiments).

**Decision logic**:
- If LINCS Δ < 0.01 AND drug_target Δ < 0.01 → "Only categorical class identity works"
- If LINCS Δ > 0.01 OR drug_target Δ > 0.01 → "Mechanistic info works if model can use it"

**Runtime**: ~15 h (3 conditions × 10 folds × 200 epochs). Can shard per fold.

**How to run**:
```bash
# All conditions, all folds
uv run python3 experiments/03_drug_feature_null/03_model_robustness/jobs/run_partC.py

# Single fold (for sharding across machines)
uv run python3 experiments/03_drug_feature_null/03_model_robustness/jobs/run_partC.py --fold 0

# Smoke test
uv run python3 experiments/03_drug_feature_null/03_model_robustness/jobs/run_partC.py --smoke
```

**Output**:
```
report/data/partC_metrics.json                    — aggregated results
report/data/fold_{k:02d}_partC_results.json       — per-fold shards
```
