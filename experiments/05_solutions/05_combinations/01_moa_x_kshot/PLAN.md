# 01_moa_x_kshot — Within-MoA training + K-shot response matching

## Research question

Do within-MoA training and K-shot response matching combine additively, or do
they exploit the same signal and saturate at the same ceiling?

Both methods independently break the cell-mean oracle (per-drug r = 0.644):
- Within-MoA training: ERK ~0.725, EGFR ~0.802
- K-shot matching (K=50): ~0.713 (averaged across all drugs)

If they exploit different information — MoA uses mechanism labels to select
training drugs, K-shot uses direct observations to identify similar drugs — then
combining them should push closer to the measurement ceiling (0.754). If they
are redundant (both ultimately identify the same nearby drugs), the combination
should equal roughly max(individual).

## Hypothesis

**Null (redundant)**: Combined per-drug r ≈ max(within-MoA alone, K-shot alone).
Both methods identify the same informative training drugs, so combining them
provides no additional signal.

**Alternative (additive)**: Combined per-drug r > max(individual), approaching
the measurement ceiling (0.754) for focus MoA classes. Within-MoA constrains the
training distribution (mechanism axis), K-shot refines predictions within that
distribution (potency axis).

Expected magnitudes for ERK MAPK signaling:
- Within-MoA alone: per-drug r ~ 0.725
- K-shot alone (K=50): per-drug r ~ 0.69 (ERK-specific)
- Combined: per-drug r ~ 0.74 if additive, ~ 0.725 if redundant
- Measurement ceiling: 0.754

## Design

**Data**: GDSC2, 233 drugs, 687 cell lines, PASO 10-fold drug-blind CV.
MoA annotations from PASO.

**Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200).

**Procedure**:
1. For each focus MoA class (ERK MAPK, EGFR signaling):
   a. Leave-one-drug-out within the MoA class.
   b. Train within-MoA Ridge on remaining same-MoA drugs.
   c. For the held-out drug, sample K anchor cells and observe their IC50.
   d. Among the same-MoA training drugs, compute response similarity on the
      K anchor cells (Pearson correlation of IC50 vectors).
   e. Final prediction = weighted blend of:
      - Within-MoA Ridge prediction (base model)
      - Response-matched reweighting of same-MoA training drugs' predictions
   f. Evaluate per-drug r on non-anchor cells.
2. K sweep: 0, 5, 10, 20, 50.
3. 10 random anchor-set draws per drug to reduce variance.
4. Compare four conditions per MoA:
   a. All-drug Ridge (baseline)
   b. Within-MoA Ridge only (K=0)
   c. All-drug Ridge + K-shot matching
   d. Within-MoA Ridge + K-shot matching (the combination)

**Blending**: For the combined method, sweep blending weight w between within-MoA
base prediction and response-matched prediction. Optimize via inner CV.

**Focus classes**: ERK MAPK signaling (~12 drugs), EGFR signaling (~8 drugs).

**Metric**: Per-drug Pearson r, macro-averaged within each MoA class, evaluated
on non-anchor cells only.

## Validation checks

- Within-MoA alone (K=0) must reproduce 02_training_distribution/01_within_moa
  results: ERK ~0.725, EGFR ~0.802.
- K-shot alone must reproduce 03_few_shot/01_response_matching results on the
  same drug subsets.
- K=0 combined must equal within-MoA alone (no observations to match on).
- Per-drug r must not exceed measurement ceiling (0.754) on average.
- Blending weight should be interpretable: if K-shot adds nothing on top of
  within-MoA, optimal w should be ~0 (all weight on the base model).
- Evaluation must exclude anchor cells (no leakage).

## Output

**`report/data/results.json`** schema:
```json
{
  "measurement_ceiling": 0.754,
  "per_moa": [
    {
      "moa": "ERK MAPK signaling",
      "n_drugs": 12,
      "conditions": {
        "all_drug_baseline": {"per_drug_r": 0.326},
        "within_moa_only": {"per_drug_r": 0.725},
        "kshot_only": {
          "k_curve": [
            {"k": 50, "per_drug_r": 0.69}
          ]
        },
        "combined": {
          "k_curve": [
            {"k": 50, "per_drug_r": 0.74, "optimal_w": 0.3}
          ]
        }
      }
    }
  ],
  "per_drug": [
    {
      "drug": "Trametinib",
      "drug_id": 1372,
      "moa": "ERK MAPK signaling",
      "all_drug_r": 0.31,
      "within_moa_r": 0.74,
      "kshot_r_k50": 0.69,
      "combined_r_k50": 0.75
    }
  ]
}
```

**`report/data/combination_results.csv`**: flat table (drug, moa, condition, k, per_drug_r).

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- MoA: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
- Prior results:
  - `02_training_distribution/01_within_moa/report/data/results.json`
  - `03_few_shot/01_response_matching/report/data/results.json`
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`

## Resources

CPU only, <30 min, --mem=32G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/05_combinations/01_moa_x_kshot/jobs/run.py
```

## Downstream use

If additive: establishes a concrete recipe (within-MoA + K-shot) that approaches
the measurement ceiling. This is the practical prescription for drug screening.

If redundant: confirms that both methods tap the same underlying signal (drug
similarity), just measured differently. The simpler method (whichever is easier to
deploy) suffices.
