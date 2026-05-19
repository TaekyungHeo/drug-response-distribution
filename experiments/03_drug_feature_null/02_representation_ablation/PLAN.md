# PLAN: Representation Ablation

## What this experiment answers

Does any drug representation class — structural, functional, mechanistic, or pharmacological
— improve within-drug cell-line ranking in the drug-blind setting?

This is the headline experiment. All representation types are evaluated under an identical
protocol in a single table, so differences in the result cannot be attributed to protocol
differences between experiments.

## Design

**Model**: Ridge regression (α=1.0)
**Cell features**: RNA PCA(550) + mutation PCA(200)
**Splits**: PASO 5-fold drug-blind CV (233 drugs, 687 cell lines)
**Minimum cell lines per drug**: 50 (drugs with fewer excluded; log n_drugs after filter)
**Primary metric**: per-drug Pearson r — reported both **unweighted mean** and **cells-weighted mean**
**Decision gate**: Δ > 0.01 over `no_drug_features` baseline (both weighting schemes must agree)
**Multiple comparisons**: Holm-Bonferroni correction across 9 non-degenerate conditions; a
  condition "crosses the gate" only if corrected p < 0.05 AND Δ > 0.01

**Power caveat (see `jobs/power_analysis.py`)**:
With K=10 folds, fold-level std ≈ 0.023, and Holm-Bonferroni over 9 conditions, the
experiment has only ~5% Monte Carlo power at the Δ=0.01 gate. The MDE at 80% power is
Δ ≈ 0.030. The null result must therefore be reported with 95% bootstrap CI on Δ:
- Primary claim: "observed Δ = 0.003 (95% CI [lo, hi])"
- Secondary claim: "effects larger than Δ ≈ 0.030 can be ruled out at 80% power"
- The gate Δ=0.01 is a clinical-relevance threshold for the drug-blind problem, not a
  detection threshold — we cannot distinguish Δ=0.003 from Δ=0.010 with this sample size.
  This limitation is acknowledged; the 7-experiment series provides convergent evidence
  that the true Δ is near-zero, not merely sub-gate.
**Splits**: pre-computed CSV files at `external/PASO/data/10_fold_data/drug_blind/
  DrugBlind_{train,test}_Fold{0-9}.csv`; no seed required to load them (FOLD_SEED=42
  documents PASO's original fold-assignment seed and is recorded for provenance only)

### Drug feature conditions

| Condition | Representation | Dim | Source |
|-----------|---------------|-----|--------|
| `no_drug` | None (cell features only) | — | baseline |
| `morgan_fp_shuffled` | Morgan FP with drug-axis permuted (drug→cell assignment broken) | 2048 | degenerate baseline: breaks drug identity |
| `random_continuous` | iid N(0,1) per drug, same dim as morgan_fp | 2048 | degenerate baseline: pure noise |
| `morgan_fp` | Morgan fingerprints, r=2, nBits=2048 | 2048 | RDKit from GDSC2 SMILES |
| `gnn` | GNN embeddings from drug-blind OmniCancerV2 | 256 | extracted from `03_model_robustness` Part B checkpoint |
| `chemberta` | ChemBERTa-zinc-base-v1 CLS, PCA(64) | 64 | Hugging Face + offline precompute |
| `chembl_targets` | ChEMBL binary protein targets | 5145 | ChEMBL v33, 5145 UniProt IDs |
| `lincs` | L1000 consensus perturbation signatures, PCA(64) | 64 | Harmonizome; 104/233 matched |
| `prism` | PRISM Repurposing pharmacological profiles, PCA(64) | 64 | DepMap portal; ~150/233 matched |
| `all_concat` | morgan_fp + chemberta + chembl_targets concatenated | 2048+64+5145 | all 233 drugs; tests combination of structural + language + mechanistic |

Notes on coverage:
- `lincs`: 104 of 233 GDSC2 drugs have L1000 signatures. **Primary metric: per-drug r on
  matched-only 104 drugs** (no zero-vector dilution). Secondary check: full 233-drug run
  with zero-vector fallback to confirm the fallback does not inflate the null.
- `prism`: same policy — matched-only (~150 drugs) is the headline; full run is secondary.
- `morgan_fp_shuffled`: drug-feature matrix rows permuted so each drug receives another
  drug's fingerprint. Tests that `morgan_fp` Δ=+0.003 is not an artifact of feature
  dimensionality alone (shuffled should recover ≈ `no_drug` r).
- `random_continuous`: iid N(0,1) vectors, one per drug. Tests same null at matched Morgan
  FP dimensionality.
- `all_concat`: concatenates morgan_fp (2048) + chemberta PCA(64) + chembl_targets (5145) for
  all 233 drugs (no coverage gap). Tests whether combining structural, language-model, and
  mechanistic representations closes the gap — addresses "each type insufficient alone" objection.
  lincs/prism excluded from concat because their partial coverage (104/~150 drugs) would reduce
  the matched drug count and confound the combination test.
- `gnn`: requires a drug-blind-trained OmniCancerV2 checkpoint. See dependency note below.

### Ridge α sensitivity check (morgan_fp only)

Run `morgan_fp` with α ∈ {0.01, 0.1, 1.0, 10, 100}. Purpose: verify α=1.0 is not
suppressing drug feature coefficients. Expected: Δ stays ≤ +0.01 across all α values.
If any α yields Δ > 0.01, α=1.0 is a confound and the canonical α must be re-chosen
via inner-fold CV before re-running all conditions.

### Dimensionality handling

Representations differ in raw dimension (64 to 5145). To prevent dimensionality from
confounding the comparison:
- All conditions use Ridge(α=1.0) — Ridge regularization is scale-invariant after
  standard normalization.
- **Continuous features** (chemberta, gnn, lincs, prism): z-scored per feature (subtract mean,
  divide by std).
- **Binary features** (`chembl_targets` 5145-dim, `morgan_fp` 2048-dim): **do not z-score**.
  Z-scoring a binary feature with prevalence p gives z=1/sqrt(p(1-p)) for positive entries —
  at p=1% this is z≈+10, amplifying rare-feature noise by 10×. Instead: drop zero-variance
  columns (features present in ≤ 1 drug across the full training set), then leave remaining
  features as {0, 1}. Ridge α penalizes coefficients in the feature-value scale; {0,1}-scale
  binary features are naturally bounded and numerically stable. Log n_features after filtering.
- `all_concat` features are normalized per-block first (using the block-appropriate method
  above), then concatenated. No re-scaling after concat.

### Secondary outputs (computed from the same fold runs)

- **Bootstrap CI on Δ**: 10,000 drug-level bootstrap samples of per-drug Δ (morgan_fp −
  no_drug) and (each rep − no_drug). Reports 95% CI for each Δ. Resampling unit is the drug
  (not the individual observation), consistent with per-drug r as primary metric.
- **Per-drug Δ histogram**: distribution of per-drug (morgan_fp r − no_drug r) across all
  test drugs, to verify the null is uniform rather than a mean of positives and negatives.
- **Per-MoA Δ stratification**: per-drug Δ broken down by GDSC2 MoA class. Verifies the null
  is not a canceling effect (e.g., kinase inhibitors +0.02, other drugs −0.02 → mean ≈ 0).
  Report: Δ mean ± std per MoA class, and fraction of classes with |Δ| > 0.01.

## Validation checks

Expected values below are from prior work on GDSC2 with Ridge and PASO splits. Deviations
are findings — log them as such rather than as failures, and investigate before concluding.

- `no_drug` per-drug r ≈ 0.631 ± 0.023 (from prior work; deviation > ±0.03 suggests data or
  split mismatch)
- `morgan_fp_shuffled` and `random_continuous` per-drug r ≈ `no_drug` (degenerate baseline
  sanity: if shuffled improves by > 0.005, the evaluation pipeline has a drug-identity leak)
- All representation conditions Δ ≤ +0.010 (expected from prior work; conditions crossing gate
  after Holm-Bonferroni are the headline finding)
- `morgan_fp` Δ bootstrap CI: ≈ [+0.002, +0.004] (from prior work; CI not overlapping zero
  confirms consistent direction, even though magnitude is below gate)
- Per-drug Δ histogram: expected unimodal near zero, ≥ 94% of drugs |Δ| ≤ 0.01
- `lincs` matched-only (104 drugs) per-drug r reported as headline; full-set value agrees within ±0.002
- Per-MoA Δ: no single MoA class drives |Δ| > 0.01
- **Minimum detectable effect (MDE)**: 80% power at α=0.05 with 233 drugs and per-drug r std
  ≈ 0.023 gives MDE ≈ +0.007. Δ in range (0.007, 0.01) is detectable but below the gate;
  report it as "statistically detectable but clinically negligible."

## Dependencies

- GDSC2 omics: `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- PASO splits: `external/PASO/data/10_fold_data/drug_blind/DrugBlind_{train,test}_Fold{0-9}.csv`
- Drug features:
  - `data/processed/morgan_fp.npy`
  - `data/processed/chembl_targets.npy` (or `.parquet`)
  - `data/processed/lincs_pca64.npy`
  - `data/processed/prism_pca64.npy`
  - `data/processed/chemberta_pca64.npy`
  - `data/processed/gnn_embeddings_256.npy` — depends on `03_model_robustness` **Part B**
    (OmniCancerV2 GCN training + embedding extraction)

Run `03_model_robustness` Part B **before** `02_representation_ablation` to produce GNN
embeddings. Part A (Transformer) can run in parallel with `02`. If `gnn_embeddings_256.npy`
is unavailable, the `gnn` condition is skipped and reported as pending.

## Memory budget (CPU, DGX Spark)

Ridge is CPU-only. Memory is dominated by the design matrix (float64, sklearn default).

| Condition | n_features | Peak memory estimate | Notes |
|-----------|-----------|---------------------|-------|
| `no_drug` | 750 | < 1 GB | cell features only |
| `morgan_fp` | 2798 | ≈ 3 GB | 128K × 2798 × 8 bytes × 2 (train+solve) |
| `chembl_targets` | 5895 | ≈ 6 GB | largest binary condition |
| `all_concat` | 8007 | ≈ 10 GB | upper bound; dominates allocation |

**SLURM allocation**: `--mem=32G` (3× peak for all_concat; safe for all conditions run
sequentially). No GPU required.

## Pre-flight checks

Before running, verify:
1. `data/processed/rna.parquet` shape: (≥ 600 cells, ≥ 18000 genes)
2. `data/processed/mutations.parquet` shape: (≥ 600 cells, ≥ 600 genes)
3. Fold CSV files present: `external/PASO/data/10_fold_data/drug_blind/DrugBlind_{train,test}_Fold{0-4}.csv`
4. `data/processed/morgan_fp.npy` shape: (233, 2048)
5. Cell × drug response matrix: 687 cells × 233 drugs after min-50-cell filter
6. 1-batch forward pass with fold 0 train data completes without error

## How to run

```bash
# Wave 1: all conditions except gnn (no prerequisites; submit immediately)
sbatch experiments/03_drug_feature_null/02_representation_ablation/jobs/sbatch_wave1.sh

# Wave 2: gnn condition only (after 03 Part B produces gnn_embeddings_256.npy)
sbatch experiments/03_drug_feature_null/02_representation_ablation/jobs/sbatch_wave2.sh
```

Expected runtime: Wave 1 < 30 min (CPU). Wave 2 < 1 min. SLURM: `--mem=32G`, no GPU, `--time=4:00:00`.

## Output

```
report/data/metrics.json          — per-condition per-drug r (mean, std, fold values, Δ, CI)
report/data/per_drug_delta.json   — per-drug Δ distribution (morgan vs no_drug)
```
