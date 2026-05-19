# PLAN: cell-only prediction ceiling

## Research question

**What is the ceiling on within-drug cell sensitivity ranking (per-drug r) using cell features alone?**

A drug-free model assigns the same prediction to every drug for a given cell line.
Its performance is entirely determined by cell-identity-driven variance in the data.
This experiment establishes that ceiling rigorously, in three stages that progressively
close alternative explanations.

**Primary metric: per-drug r** (macro-averaged Pearson r within each drug, ≥5 test
samples). This follows directly from `01_global_vs_perdrug`, which established that
global Pearson r conflates between-drug potency ranking (irrelevant to the clinical task)
with within-drug cell ranking (the actual target). Here, the structural argument is even
stronger: a drug-free model has *exactly zero* between-drug discriminative ability by
construction, so global r is penalized for a signal the model was never designed to produce.
Reporting the ceiling in global r would understate the drug-free model's true within-drug
ranking ability. Global r is reported as a secondary diagnostic only.

Scope distinction: `03_cell_representation/02_cell_ceiling` measures the ceiling for
a model WITH drug features (OmniCancerV1, cell-blind per-drug r = 0.499). This
experiment measures the ceiling WITHOUT drug features — a strictly lower quantity that
answers a different question: how much is cell identity worth, with no drug knowledge?

---

## Relation to existing results

A prior run (`results/run_20260519_000000/`) evaluated RNA-only, Concatenation, and
Late fusion models for 50 epochs with **global Pearson r as the primary metric**.
Those results are **discarded** for two reasons:

1. Per-drug r is now the primary metric (established by `01_global_vs_perdrug`). The
   prior run did not compute per-drug r, so its numbers cannot be compared to the new
   plan's primary output.
2. 50 epochs with no convergence tracking is insufficient; the new plan uses max 300
   epochs with early stopping and learning-curve recording.

The prior `jobs/run_baselines.py` and `report/data/metrics.json` are kept in place for
git history but are not used in any downstream analysis. All fresh results go to
`results/<stage>/<run_id>/`.

---

## Data

**Source**: GDSC2 IC₅₀ measurements.

| | value |
|--|--|
| Cell lines with all 5 omics | 597 |
| Drugs | 286 |
| (cell, drug) pairs | 151,515 of 170,742 possible (11.3% missing) |
| Response variable | ln_IC₅₀; range [−8.77, 13.85], mean 2.84, std 2.76 |

**Omics features** (rows = cell lines, all zero-missing):

| Modality | Features |
|----------|----------|
| RNA-seq | 19,193 |
| Somatic mutations | 12,301 |
| Copy number variation | 38,590 |
| Metabolomics | 225 |
| RPPA | 214 |
| **Total concat** | **70,523** |

**Splits**: three protocols from `src/data/splits.py`, 70 / 10 / 20 train / val / test.

| Split | Description |
|-------|-------------|
| Mixed-set | (cell, drug) pairs assigned randomly |
| Cell-blind | Cell lines assigned as whole units — test cells unseen during training |
| Drug-blind | Drugs assigned as whole units — test drugs unseen during training |

---

## Stage 0: Analytical upper bounds (no training required)

Compute before any model. These bounds cannot be exceeded by any drug-free model;
if a trained model exceeds them, the oracle implementation is wrong.

**Cell-mean oracle**: for each cell c, predict its mean response ȳ_c.
The oracle target is computed from the same split's data:

| Split | Oracle target for cell c |
|-------|--------------------------|
| Mixed-set | Mean ln_IC₅₀ over c's *training* drugs |
| Cell-blind | Mean ln_IC₅₀ over c's *test* drugs (**deliberate label leak** — oracle uses test labels to establish an unattainable upper bound; no non-leaking model can reach this) |
| Drug-blind | Mean ln_IC₅₀ over c's *training* drugs |

The cell-mean oracle upper bound is the **per-drug r** between {ȳ_c repeated per test drug}
and actual test responses. Global r is also computed but is secondary.

**Sanity baselines** (expected behavior pre-registered before running):
- Global mean predictor: predict grand mean μ for every pair.
  Expected global r < 0.01; expected per-drug r ≈ 0 (same constant within each drug →
  zero within-drug variance in predictions → Pearson undefined → treat as 0.0).
- Per-drug mean predictor: for each drug d, predict its training mean (tests whether
  drug-identity signal accidentally leaks into a supposedly drug-free evaluation).
  Expected global r: high (correctly ranks drugs by potency). Expected per-drug r ≈ 0
  (same constant per drug → undefined → 0.0 by policy). Any non-zero per-drug r here
  indicates evaluation leakage.

**Per-cell r and per-drug r** are computed for the oracle to give upper bounds
on all six primary ceilings (3 splits × 2 metrics). Per-drug r is the primary ceiling;
per-cell r is secondary.

Script: `jobs/compute_oracles.py`  
Runtime: < 1 min

---

## Stage 1: Ridge regression

Ridge regression is architecture-free and tests whether the RNA-seq baseline is a data
limitation or a model limitation. If Ridge ≥ MLP on per-drug r, the MLP was not the
bottleneck; if Ridge < MLP, the nonlinearity matters.

**Protocol**:
- Input: RNA-seq only (19,193 features), z-score normalized on train set
- Alpha grid: [0.01, 0.1, 1, 10, 100, 1000] — chosen by best val-set **per-drug r**
- Report: per-drug r (primary), global r (secondary), per-cell r (min 5 drugs); all
  macro-averaged with ≥5 samples threshold
- 5-fold CV on all three splits to estimate fold-to-fold variance

Script: `jobs/run_ridge.py`  
Runtime: < 5 min

---

## Stage 2: Fixed-capacity MLP — capacity sweep and modality ablation

The original experiment varied modality count and architecture capacity simultaneously,
confounding the two effects. This stage separates them.

**Sub-experiment A — Capacity sweep (RNA-only):**
- Architectures: Small [128→64→1] / Medium [512→256→64→1] / Large [2048→512→128→1]
- Sweep: dropout [0.1, 0.3, 0.5] × weight_decay [0, 1e-4, 1e-3]
- Training: LR cosine [1e-3 → 1e-5], batch_size=512, max_epochs=300,
  early stopping patience=30 on val per-drug r; best checkpoint restored.
- Learning curves (val per-drug r per epoch) stored for every run.
- **Convergence check**: if best checkpoint is at epoch ≥ 270, flag run as
  potentially unconverged — do not use its result without re-running with more epochs.
  (Threshold = max_epochs − patience = 300 − 30 = 270.)
- Best config per capacity level selected by val-set **per-drug r** on mixed-set split.
- Each winning config evaluated on all three splits.
- If Large ≈ Medium on per-drug r → capacity is not the bottleneck.

**Sub-experiment B — Modality ablation (Medium capacity, best config from A):**
- Variants: RNA-only → +mutations → +CNV → +metabolomics → all-5-omics
- Fixed architecture [512→256→64→1] for all variants.
- Training: LR cosine [1e-3 → 1e-5], batch_size=512, max_epochs=300,
  early stopping patience=30 on val per-drug r; best checkpoint restored.
- Learning curves (val per-drug r per epoch) stored for every run.
- **Convergence check**: same as Sub-experiment A — flag if best epoch ≥ 270.

Script: `jobs/run_mlp_sweep.py`  
Runtime: ~1–2 h on GB10 (CUDA)

---

## Stage 3: Cell-blind regularization sweep

Cell-blind val per-drug r is expected to peak early (overfitting to training cells) and
then decline. This stage asks: is overfitting the binding constraint for cell-blind
per-drug r, or is the bottleneck the data?

**Protocol**:
- RNA-only MLP, fixed architecture [512→256→64→1]
- Grid: dropout [0.2, 0.3, 0.5] × weight decay [1e-3, 1e-2, 5e-2]
- Training: LR cosine [1e-3 → 1e-5], batch_size=512, max_epochs=300,
  early stopping patience=30 on val per-drug r.
- Learning curves (val per-drug r per epoch) stored for every config.
- Report: best val **per-drug r** and the epoch at which it occurs for each config.
- **Convergence check**: flag any config whose best checkpoint is at epoch ≥ 270.
- If no config improves beyond baseline cell-blind per-drug r, regularization is not the
  binding constraint — the bottleneck is the data or the model capacity.

Script: `jobs/run_cellblind_reg_sweep.py`  
Runtime: ~30 min

---

## Primary results table

Report all six ceilings for every model/split combination:

| Model | MS per-drug r | MS global r | CB per-drug r | CB global r | DB per-drug r | DB global r |
|-------|--------------|------------|--------------|------------|--------------|------------|
| Cell-mean oracle | — | — | — | — | — | — |
| Global mean | — | — | — | — | — | — |
| Ridge (RNA-only) | — | — | — | — | — | — |
| MLP RNA-only (best) | — | — | — | — | — | — |
| MLP all-5-omics (best) | — | — | — | — | — | — |

MS = mixed-set, CB = cell-blind, DB = drug-blind.

---

## Validation checks

- Cell-mean oracle per-drug r ≥ all trained models' per-drug r (else oracle is wrong)
- Global mean predictor: global r < 0.01; per-drug r = 0.0 by policy (undefined → 0 imputation)
- Per-drug mean predictor: per-drug r = 0.0 by policy; any non-zero value indicates leakage
- Ridge val per-drug r ≥ MLP val per-drug r (if not, Ridge alpha sweep is wrong)
- Cell-blind best val per-drug r occurs at epoch ≤ 30 for default dropout; regularization
  sweep should not improve beyond that ceiling unless overfitting was the binding constraint
- Per-drug r (primary) and per-cell r (secondary) both reported for all six split combinations
- Global r reported as supplementary diagnostic for all combinations

---

## Dependencies

**Data** (all stages read directly from `data/processed/`):
- `{rna,mutations,cnv,metabolomics,rppa,drug_response}.parquet`
- `overlap_cell_lines.parquet`

**Code**:
- `src/data/splits.py` — `mixed_set_split`, `cell_blind_split`, `drug_blind_split`
- `src/evaluation/metrics.py` — `evaluate`
- `src/evaluation/per_drug.py` — `per_drug_r`, `mean_per_drug_r`
- `src/utils/ridge.py` — `safe_fit_scaler`

**Inter-stage runtime dependency** (only one):
- Stage 2B reads `results/stage2/capacity_sweep_best_configs.json` written by Stage 2A.
  All other stages are independent of each other and can run in parallel.

**Execution order** (verified by inspecting script imports):

```
Stage 0 ──┐
Stage 1   ├── all independent, submit simultaneously
Stage 2A  │
Stage 3 ──┘
           ↓ (Stage 2A writes capacity_sweep_best_configs.json)
Stage 2B ×5  (parallel, each variant reads that file)
```

**Relation to 02_metric_selection**: none. This experiment hardcodes per-drug Pearson r
as primary metric, citing `01_global_vs_perdrug`. `02_metric_selection` is expected to
confirm Pearson ≈ Spearman at n ≈ 22; even in the unlikely event it does not, the
numbers would be nearly identical. Both experiments run in parallel.

---

## How to run

```bash
# Submit all stages with correct dependencies (only Stage 2B waits for 2A):
bash experiments/01_metric_decomposition/03_baselines/jobs/launch.sh
```

Results for each stage are written to `results/stage{0,1,2,3}/` and aggregated
to `report/data/metrics.json` for report rendering.
