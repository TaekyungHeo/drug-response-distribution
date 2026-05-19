# PLAN: Dataset Robustness

## What this experiment answers

`02_representation_ablation` uses GDSC2 (233 drugs, 687 cell lines, IC₅₀). A skeptic can
argue: "GDSC2 is too small — 233 drugs may be insufficient for a model to learn drug-structure
→ sensitivity transfer. With more drugs and pharmacological diversity, structural features
might provide a meaningful signal."

This experiment replicates the Morgan FP vs no_drug ablation on the PRISM Repurposing dataset:
657 drugs (3× GDSC2), 477 cell lines, viability-based (AUC), a completely different
experimental platform. If the null holds at 3× the drug count and a different assay type,
it is a fundamental property of the drug-blind prediction problem, not a GDSC2 artifact.

## Design

### Primary: PRISM Repurposing

| Setting | Value |
|---------|-------|
| Dataset | PRISM Repurposing (DepMap portal) |
| Drugs | 657 with valid Morgan fingerprints |
| Cell lines | 477 |
| Response | viability AUC |
| Model | Ridge(α=1.0) |
| Cell features | RNA PCA(550) + mutation PCA(200) (PRISM cell lines ∩ CCLE RNA) |
| Conditions | `morgan_fp` vs `no_drug` |
| Splits | 5-fold drug-blind CV (stratified by drug count per fold) |
| Metric | Per-drug Pearson r |

### Secondary: Transformer on PRISM

Run OmniCancerV1 with Morgan FP vs no_drug on PRISM to verify model class robustness
on this dataset as well (parallels `03_model_robustness` but on PRISM).

## Why PRISM is the right choice

- 657 drugs (3× GDSC2)
- Different assay platform (viability vs IC₅₀) — tests cross-platform generality
- Has already been used in the field to test DRP generalization
- Substantial overlap with CCLE for cell features

## Validation checks

- Ridge `no_drug` per-drug r ≈ 0.330 ± 0.006 (lower than GDSC2 due to assay noise)
- Morgan FP Δ ≤ +0.003 (null replicates at 3× drug count)
- Transformer `no_drug` per-drug r in same range
- n_drugs = 657 (verified after SMILES filtering)

## Known limitations

1. **Cell features**: CCLE RNA shared with GDSC2 pipeline. The cell-side feature distribution
   is nearly identical to the GDSC2 experiments. This experiment varies the drug-axis (more
   drugs, different assay platform) while holding the cell axis nearly constant; it does not
   constitute an independent test of the cell representation.

2. **IC₅₀ vs AUC comparability**: GDSC2 reports IC₅₀ (concentration at half-maximal effect);
   PRISM reports viability AUC. The raw per-drug r magnitudes are NOT directly comparable
   between datasets (AUC has different noise characteristics and range than IC₅₀). Only the
   **null pattern** — whether Δ > 0.01 — is compared across datasets. Expected per-drug r
   values for PRISM will differ from GDSC2 baselines (PRISM no_drug ≈ 0.330 vs GDSC2 ≈ 0.631);
   this does not indicate a problem.

## Prerequisites

- PRISM Repurposing data: `data/external/prism/repurposing_secondary_screen_dose_response.csv`
  (downloaded from DepMap portal if absent)
- CCLE RNA-seq aligned to PRISM cell lines: `data/processed/rna.parquet` (shared with GDSC2)
- `data/processed/morgan_fp_prism.npy` (Morgan FP for PRISM 657 drugs)

## Random seeds (Transformer only)

- `FOLD_SEED=42`
- `MODEL_SEED=0`
- `BATCH_SEED=0`

## Memory budget — Transformer only (DGX Spark, 128 GB unified)

| Component | Estimate | Notes |
|-----------|---------|-------|
| Model parameters (21M) | ≈ 0.08 GB | bf16 |
| PRISM pairs: 657 drugs × 477 cells = 313K pairs/fold | ≈ 0.6 GB activations at batch=256 | larger than GDSC2 |
| Total peak | **< 6 GB** | safe |

- Mixed precision: **bf16**
- Batch size: 256 pairs; reduce to 128 if OOM

## Required telemetry — Transformer only (per epoch)

```
### Training dynamics
- train_loss, val_loss
- train_per_drug_r, val_per_drug_r
- learning_rate, grad_norm_pre_clip, epoch_time_s

### System
- gpu_memory_peak_gb

### Final (per fold)
- best_epoch, val_per_drug_r_at_best, test_per_drug_r_at_best

### Outputs to disk
- logs/fold{k}_{condition}_epoch_metrics.parquet
- logs/fold{k}_{condition}_test_predictions.parquet
```

## Distribution check (PRISM preprocessing)

Before running, verify:
- PRISM AUC distribution: mean ≈ 0.8, flag any drug with > 10% failed wells
- Exclude drugs with < 50 cell lines tested (minimum-cell-line filter)
- Log n_drugs after filtering (expected ≈ 630–657)

## Pre-flight checks

Before running, verify:
1. `data/external/prism/repurposing_secondary_screen_dose_response.csv` present and parseable
2. PRISM AUC distribution: mean ≈ 0.8; flag and exclude drugs with > 10% failed wells
3. n_drugs after filtering ≈ 630–657 (log exact count)
4. `data/processed/rna.parquet` cell-line index overlaps with PRISM cell lines: verify ≥ 400 shared
5. `data/processed/morgan_fp_prism.npy` shape: (n_prism_drugs, 2048)
6. Transformer only: 1-batch forward pass with fold 0 train data completes before full GPU run

## How to run

```bash
# Ridge (CPU)
sbatch experiments/03_drug_feature_null/05_dataset_robustness/jobs/sbatch_ridge.sh

# Transformer (GPU)
sbatch experiments/03_drug_feature_null/05_dataset_robustness/jobs/sbatch_transformer.sh
```

Expected runtime: Ridge < 30 min (SLURM: `--mem=32G`, no GPU). Transformer ≈ 2 h (SLURM: `--mem=64G`, `--gres=gpu:1`, `--time=4:00:00`).

## Output

```
report/data/metrics.json   — per-drug r (mean, std, fold values, Δ) for Ridge and Transformer
                             n_drugs, n_cells confirmed
logs/                      — Transformer telemetry parquets
```
