# 01_moa_performance — Per-MoA model performance landscape

## Research question

How does per-drug model performance vary across MoA classes under all-drug training?

## Hypothesis

MoA classes differ systematically in per-drug r. Mechanistically coherent classes
(Mitosis, Cell cycle) where cell sensitivity is dominated by proliferation rate
should be easy (high per-drug r), because the cell-mean signal aligns with the
drug's mechanism. Classes targeting heterogeneous pathways (ERK MAPK, Apoptosis)
should be hard (low per-drug r), because the cell-mean baseline carries less
drug-specific information.

## Design

**Data**: GDSC2, 233 drugs, 687 cell lines, PASO 10-fold drug-blind CV.
MoA annotations: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
(column: `Target Pathway`).

**Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200).
This is the standard all-drug model from prior experiments.

**Procedure**:
1. Run Ridge across all 10 PASO drug-blind folds (or load cached predictions).
2. Compute per-drug Pearson r for each drug (≥5 test samples per drug per fold).
3. Average each drug's r across folds to get a single per-drug r estimate.
4. Join with MoA annotations; group by `Target Pathway`.
5. Report per-MoA: mean r, std r, n_drugs, sorted descending by mean r.
6. Flag systematically hard classes (mean r < 0.3) and easy classes (mean r > 0.5).

**Metric**: Per-drug Pearson r, macro-averaged within each MoA class.

## Validation checks

- Grand mean across all drugs should match the known all-drug baseline (~0.38).
- Unclassified drugs excluded from MoA grouping (but reported separately).
- MoA classes with <3 drugs flagged as unreliable estimates.
- Per-drug r distribution should be continuous, not bimodal (bimodality would
  suggest a confound like sample-size dependence rather than biology).

## Output

**`report/data/results.json`** schema:
```json
{
  "overall_mean_r": 0.38,
  "n_drugs": 233,
  "per_moa": [
    {
      "moa": "Mitosis",
      "mean_r": 0.55,
      "std_r": 0.12,
      "n_drugs": 8,
      "drugs": ["Docetaxel", "..."]
    }
  ],
  "per_drug": [
    {
      "drug": "Docetaxel",
      "drug_id": 1007,
      "moa": "Mitosis",
      "mean_r": 0.62,
      "n_folds": 10
    }
  ]
}
```

**`report/data/moa_performance.csv`**: flat table (drug, moa, mean_r, std_r, n_folds).

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- MoA: `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv`
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`

## Resources

CPU only, <5 min, --mem=16G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/01_diagnosis/01_moa_performance/jobs/run.py
```

## Downstream use

Results motivate `02_training_distribution`: if MoA predicts difficulty, then
MoA-stratified training (restricting training to same-MoA drugs) may improve
the hard classes. Also feeds into `02_moa_ceiling` to compare observed per-drug r
against the within-MoA biological ceiling.
