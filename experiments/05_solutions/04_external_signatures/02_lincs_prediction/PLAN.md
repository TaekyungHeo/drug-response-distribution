# 02_lincs_prediction — Can LINCS signatures be predicted from drug structure?

## Research question

Can LINCS L1000 transcriptional signatures be predicted from Morgan fingerprints?

This is a gate experiment that closes the structure-function loop:
- 03_drug_feature_null: structural features (Morgan FP) don't help per-drug r.
- 04_external_signatures/01_lincs: LINCS helps global r but not per-drug r.
- This experiment: can structure predict LINCS?

If yes, then LINCS's global-r benefit could be obtained from structure alone
(without measuring signatures), and the fact that structure doesn't help in
03_drug_feature_null is puzzling. If no, then LINCS is genuinely novel information
not recoverable from chemical structure, which explains why structural features
fail — they simply don't encode the functional information that LINCS captures.

## Hypothesis

**Null**: Morgan FP can predict LINCS PCA(64) with R² > 0.1, indicating that
structural features encode some transcriptional effect information.

**Alternative**: R² ≤ 0 (at or below chance), confirming that drug structure
cannot predict transcriptional effect. This closes the explanatory loop:
structure doesn't help (03) because it can't encode mechanism (this experiment),
and LINCS helps global r (01_lincs) because it directly measures what structure
cannot capture.

Expected: R² < 0 (Ridge regression from Morgan FP to LINCS PCA components
performs worse than predicting the mean).

## Design

**Data**: The ~104 drugs that appear in both GDSC2 and LINCS L1000.

**Input**: Morgan fingerprints (2048-bit, radius 2) for each drug.

**Target**: LINCS L1000 consensus signature, PCA-reduced to 64 dimensions.
Each PCA component is a separate regression target.

**Model**: Ridge regression, Morgan FP (2048) → LINCS PCA(64).

**Procedure**:
1. Load Morgan FP and LINCS PCA(64) for the ~104 overlapping drugs.
2. Leave-one-drug-out cross-validation:
   a. For each held-out drug, train Ridge on the remaining ~103 drugs.
   b. Predict the held-out drug's LINCS PCA(64) vector.
   c. Compute per-component R² and overall R² (across all components).
3. Also compute cosine similarity between predicted and true LINCS vectors.

**Alpha sweep**: Ridge alpha in {0.01, 0.1, 1.0, 10, 100, 1000} with inner CV.

**Control — permuted fingerprints**: Shuffle Morgan FP across drugs (breaking
structure-signature correspondence). This must yield R² ≈ 0 or negative,
confirming that any positive R² in the real condition is not an artifact.

**Metric**: R² (coefficient of determination) on held-out drugs, averaged across
PCA components. Also report per-component R² to check if any individual
components are predictable.

## Validation checks

- Number of drugs must match the overlap count from 01_lincs (~104).
- Permuted control must yield R² ≤ 0.
- Per-component R² distribution should be reported — even if overall R² is
  negative, some individual components might be weakly predictable.
- Morgan FP should have non-trivial variance (not all drugs identical).
- LINCS PCA(64) should have non-trivial variance per component.

## Output

**`report/data/results.json`** schema:
```json
{
  "n_drugs": 104,
  "morgan_fp_dim": 2048,
  "lincs_pca_dim": 64,
  "best_alpha": 10.0,
  "overall_r2": -0.15,
  "mean_cosine_sim": 0.05,
  "permuted_r2": -0.20,
  "per_component": [
    {
      "pc": 0,
      "r2": -0.08,
      "variance_explained_pct": 15.2
    }
  ]
}
```

**`report/data/prediction_results.csv`**: flat table (drug, drug_id, true_pc0, pred_pc0, ..., cosine_sim).

## Dependencies

- LINCS: Same preprocessed signatures as 01_lincs
- Morgan FP: RDKit Morgan fingerprints for GDSC2 drugs
- Drug mapping: GDSC2 drug ID to SMILES/InChI
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`

## Resources

CPU only, <5 min, --mem=8G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/04_external_signatures/02_lincs_prediction/jobs/run.py
```

## Downstream use

This experiment closes the structure-function-prediction triangle:
- Structure → drug response: doesn't help per-drug r (03_drug_feature_null)
- LINCS → drug response: helps global r, not per-drug r (01_lincs)
- Structure → LINCS: expected R² < 0 (this experiment)

The conclusion: functional signatures (LINCS) contain information that structure
does not, but even this information is drug-level and cannot break the cell-mean
oracle for per-drug r. The only paths to per-drug r > 0.644 remain training
distribution (within-MoA) and direct observation (K-shot).
