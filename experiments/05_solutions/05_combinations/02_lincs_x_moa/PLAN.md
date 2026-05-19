# 02_lincs_x_moa — LINCS signatures + within-MoA training

## Research question

Do LINCS drug features and within-MoA training combine to improve BOTH global r
AND per-drug r simultaneously?

These two interventions operate on orthogonal axes:
- LINCS improves global r (between-drug scale) but not per-drug r.
- Within-MoA training improves per-drug r (within-drug ranking) via training
  distribution selection.

If they are truly orthogonal, combining them should yield improvements on both
axes simultaneously — global r from LINCS and per-drug r from within-MoA — with
no interference.

## Hypothesis

**Null**: The combination yields no benefit beyond the individual methods.
LINCS features are redundant when training is already restricted to same-MoA
drugs (because same-MoA drugs already have similar LINCS profiles).

**Alternative**: The combination improves BOTH metrics simultaneously:
- Global r: improved by LINCS (between-drug potency encoding)
- Per-drug r: improved by within-MoA (mechanism-specific cell features)

The effects are additive because they target different variance components.

Expected magnitudes (on LINCS-covered drugs within focus MoAs):
- All-drug, no LINCS (baseline): global r ~ 0.48, per-drug r ~ 0.33
- Within-MoA only: global r ~ 0.50, per-drug r ~ 0.72
- LINCS only: global r ~ 0.65, per-drug r ~ 0.33
- Combined: global r ~ 0.67, per-drug r ~ 0.72

## Design

**Data**: GDSC2, restricted to drugs that are BOTH in a focus MoA class AND have
LINCS L1000 signatures. 687 cell lines. PASO 10-fold drug-blind CV.

**Drug features**: LINCS L1000 PCA(64), same as 04_external_signatures/01_lincs.

**Model**: Ridge(alpha=1.0), cell features = RNA PCA(550) + mutation PCA(200),
drug features = LINCS PCA(64) (when used).

**Procedure**:
1. Identify drugs that are in both LINCS and a focus MoA class.
   Report overlap size per MoA (e.g., ERK: N/12 drugs have LINCS).
2. For each focus MoA class with sufficient LINCS-covered drugs (>=3):
   a. Run four conditions, all on the same drug subset:
      - **all_drug_no_lincs**: All-drug training, cell features only.
      - **within_moa_no_lincs**: Within-MoA training, cell features only.
      - **all_drug_lincs**: All-drug training, cell + LINCS features.
      - **within_moa_lincs**: Within-MoA training, cell + LINCS features.
   b. Leave-one-drug-out within the MoA class for within-MoA conditions.
      Use PASO drug-blind CV for all-drug conditions.
   c. For each condition, compute global r and per-drug r.
3. Report the 2x2 factorial table (MoA x LINCS) for each metric.

**Focus classes**: ERK MAPK signaling, EGFR signaling (contingent on having
>=3 LINCS-covered drugs in each class).

**Metric**: Global Pearson r AND per-drug Pearson r, both computed on the
LINCS-covered drugs within each MoA class.

## Validation checks

- within_moa_no_lincs per-drug r must match 02_training_distribution/01_within_moa
  results for the same drug subset (ERK ~0.725, EGFR ~0.802, restricted to
  LINCS-covered drugs).
- all_drug_lincs global r improvement must be consistent with
  04_external_signatures/01_lincs results on the same drug subset.
- all_drug_no_lincs is the lowest-performing condition on both metrics.
- LINCS should not degrade per-drug r (delta >= -0.01 for any condition).
- Within-MoA should not degrade global r.
- If a MoA class has <3 LINCS-covered drugs, exclude it and report why.
- All four conditions must be evaluated on the EXACT same drug subset.

## Output

**`report/data/results.json`** schema:
```json
{
  "per_moa": [
    {
      "moa": "ERK MAPK signaling",
      "n_drugs_total": 12,
      "n_drugs_lincs": 8,
      "factorial": {
        "all_drug_no_lincs": {"global_r": 0.48, "per_drug_r": 0.33},
        "within_moa_no_lincs": {"global_r": 0.50, "per_drug_r": 0.72},
        "all_drug_lincs": {"global_r": 0.65, "per_drug_r": 0.33},
        "within_moa_lincs": {"global_r": 0.67, "per_drug_r": 0.72}
      }
    }
  ],
  "per_drug": [
    {
      "drug": "Trametinib",
      "drug_id": 1372,
      "moa": "ERK MAPK signaling",
      "has_lincs": true,
      "all_drug_no_lincs_r": 0.31,
      "within_moa_no_lincs_r": 0.74,
      "all_drug_lincs_r": 0.32,
      "within_moa_lincs_r": 0.74
    }
  ]
}
```

**`report/data/factorial_results.csv`**: flat table (moa, condition, global_r, per_drug_r, n_drugs).

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- MoA: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
- LINCS: Same preprocessed signatures as 04_external_signatures/01_lincs
- Prior results:
  - `02_training_distribution/01_within_moa/report/data/results.json`
  - `04_external_signatures/01_lincs/report/data/results.json`
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`

## Resources

CPU only, <15 min, --mem=16G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/05_combinations/02_lincs_x_moa/jobs/run.py
```

## Downstream use

If orthogonal: establishes that global r and per-drug r are independently
improvable — LINCS for scale, within-MoA for ranking. This supports a practical
two-stage predictor: use LINCS to place drugs on the right scale, use within-MoA
to rank cells correctly.

If redundant: within-MoA training already captures the mechanism information that
LINCS provides, making LINCS unnecessary when MoA labels are available. This
would simplify the practical recipe to: within-MoA training + K-shot (from
01_moa_x_kshot).
