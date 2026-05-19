# 03_onehot_control — MoA label as feature (representation control)

## Research question

Does adding MoA class identity as a one-hot feature to the Ridge input improve
per-drug r?

This is the critical control for 01_within_moa. Both experiments give the model
access to MoA information, but through different mechanisms:
- 01_within_moa: MoA determines which training samples the model sees (distribution)
- 03_onehot_control: MoA is an input feature the model can use (representation)

If MoA-as-feature helps, the within-MoA gains are simply "MoA information helps
prediction." If MoA-as-feature does NOT help but within-MoA training does, this
proves the mechanism is training distribution, not representation. This parallels
the finding from 03_drug_feature_null: drug features as representation do not
improve per-drug r.

## Hypothesis

**Null**: MoA one-hot features improve per-drug r by an amount comparable to
within-MoA training (delta >= +0.05). The model uses MoA identity to learn
drug-class-specific cell patterns.

**Alternative**: MoA one-hot features yield delta <= +0.002 over the all-drug
baseline. Ridge cannot use a drug-level feature (constant across all cells for a
given drug) to improve per-drug r, because per-drug r measures within-drug cell
ranking, which is orthogonal to drug identity. This is the same mechanism as
03_drug_feature_null.

**Why the alternative should hold**: Per-drug Pearson r is invariant to
per-drug location shifts. A feature that is constant for all cells treated with
the same drug (like MoA one-hot) can only shift the predicted mean for that drug,
not change the cell ranking. Ridge with such a feature will improve global r
(by capturing drug-level variance) but not per-drug r.

## Design

**Data**: GDSC2, 233 drugs, 687 cell lines, PASO 10-fold drug-blind CV.
MoA annotations: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`.

**Model**: Ridge(alpha=1.0), features = RNA PCA(550) + mutation PCA(200) + MoA one-hot.

**Procedure**:
1. One-hot encode MoA class for each drug. Drugs with no MoA annotation get
   all-zeros. Append as additional features to the cell feature matrix
   (each training sample inherits the one-hot vector of its drug).
2. Run standard PASO 10-fold drug-blind CV with the augmented feature matrix.
3. Compute per-drug Pearson r for each drug.
4. Compare against the all-drug baseline (same model, no MoA features).

**Also report global Pearson r**: expected to improve slightly (MoA features
capture drug-level mean differences), confirming the features are being used --
just not for the metric that matters.

**Metric**: Per-drug Pearson r (primary), global Pearson r (diagnostic).

## Validation checks

- Feature matrix shape: n_features = 550 + 200 + n_moa_classes (~15-20).
- Verify MoA one-hot columns are non-zero in the training data (features are
  actually present, not zeroed out by preprocessing).
- All-drug baseline per-drug r must match known value (~0.38 mean, 0.644 oracle).
- Global r should improve vs. baseline (proves the model is using MoA features
  for something -- just not per-drug ranking).
- If per-drug r delta > +0.01, investigate: likely a bug in feature construction
  (e.g., MoA features leaking cell-level information).

## Output

**`report/data/results.json`** schema:
```json
{
  "baseline": {
    "mean_per_drug_r": 0.38,
    "global_r": 0.72,
    "n_drugs": 233
  },
  "with_moa_onehot": {
    "mean_per_drug_r": 0.382,
    "global_r": 0.74,
    "n_drugs": 233
  },
  "delta_per_drug_r": 0.002,
  "delta_global_r": 0.02,
  "conclusion": "MoA as representation does not improve per-drug r"
}
```

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- MoA: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`

## Resources

CPU only, <15 min, --mem=32G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/02_training_distribution/03_onehot_control/jobs/run.py
```

## Downstream use

Combined with 01_within_moa, this establishes the central claim of the project:
**representation != distribution.** Same MoA information, opposite outcomes.
This distinction is the key insight motivating all of 05_solutions: the bottleneck
is not what the model knows about the drug, but which drugs the model trains on.
