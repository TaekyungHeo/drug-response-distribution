# 01_within_moa — Strict within-MoA training

## Research question

Does restricting training to same-MoA drugs break the cell-mean oracle ceiling
(per-drug r = 0.644)?

Under all-drug training, the model learns a cell-mean signal that is
drug-agnostic. By training only on drugs that share a mechanism, the model should
learn cell features predictive of sensitivity to *that mechanism*, not just
general sensitivity.

## Hypothesis

**Null**: Within-MoA training yields per-drug r no higher than the all-drug
cell-mean oracle (0.644). The cell-mean signal dominates regardless of training
composition.

**Alternative**: Within-MoA training substantially exceeds 0.644 for MoA classes
with high profile concordance (ERK MAPK, EGFR), because restricting training
drugs forces the model to learn mechanism-specific cell features.

Expected magnitudes (from 01_diagnosis results):
- ERK MAPK signaling: all-drug ~0.326 -> within-MoA ~0.725
- EGFR signaling: all-drug ~0.405 -> within-MoA ~0.802
- Mitosis (easy control): all-drug already high, within-MoA similar or slightly lower
- Cell cycle (easy control): same -- within-MoA should not help much

## Design

**Data**: GDSC2, 233 drugs, 687 cell lines, PASO 10-fold drug-blind CV.
MoA annotations: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
(column: `Target Pathway`).

**Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200).

**Procedure**:
1. For each MoA class with >=3 drugs:
   a. Select all drugs in that class.
   b. Leave-one-drug-out within the class: for each held-out drug, train Ridge
      on all remaining same-MoA drugs only.
   c. Predict the held-out drug's IC50 across all cell lines.
   d. Compute per-drug Pearson r for the held-out drug.
2. Average per-drug r within each MoA class.
3. Compare against the all-drug baseline per-drug r for the same drugs
   (from 01_diagnosis/01_moa_performance).

**Focus classes**: ERK MAPK signaling, EGFR signaling (hard under all-drug),
Mitosis, Cell cycle (easy controls).

**Metric**: Per-drug Pearson r, macro-averaged within each MoA class.

## Validation checks

- All-drug baseline per-drug r for the same drugs must match 01_diagnosis values.
- Per-drug r for each drug must be computed on >=5 test cell lines.
- MoA classes with <3 drugs excluded (leave-one-drug-out requires >=2 training drugs).
- Cell-mean oracle per-drug r within each MoA should be computable as a sanity
  check: within-MoA training should approach but not exceed the within-MoA
  measurement ceiling.
- Easy classes (Mitosis, Cell cycle) should show minimal change from all-drug
  baseline; large gains there would suggest the effect is not mechanism-specific.

## Output

**`report/data/results.json`** schema:
```json
{
  "overall": {
    "all_drug_mean_r": 0.38,
    "within_moa_mean_r": 0.52,
    "n_moa_classes": 15,
    "n_drugs": 180
  },
  "per_moa": [
    {
      "moa": "ERK MAPK signaling",
      "all_drug_mean_r": 0.326,
      "within_moa_mean_r": 0.725,
      "delta": 0.399,
      "n_drugs": 12,
      "drugs": ["Trametinib", "..."]
    }
  ],
  "per_drug": [
    {
      "drug": "Trametinib",
      "drug_id": 1372,
      "moa": "ERK MAPK signaling",
      "all_drug_r": 0.31,
      "within_moa_r": 0.74
    }
  ]
}
```

**`report/data/within_moa_results.csv`**: flat table (drug, moa, all_drug_r, within_moa_r, delta).

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- MoA: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
- Baseline: `experiments/05_solutions/01_diagnosis/01_moa_performance/report/data/results.json`
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`

## Resources

CPU only, <30 min, --mem=32G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/02_training_distribution/01_within_moa/jobs/run.py
```

## Downstream use

If within-MoA training breaks 0.644, this establishes training distribution as the
first known mechanism for exceeding the cell-mean oracle. Feeds into:
- `02_moa_weighted` (soft version for practical deployment)
- `03_onehot_control` (proves mechanism is distribution, not representation)
- `05_combinations/01_moa_x_kshot` (combining with few-shot)
