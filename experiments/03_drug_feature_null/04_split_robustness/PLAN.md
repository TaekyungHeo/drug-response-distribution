# PLAN: Split Robustness

## What this experiment answers

`02_representation_ablation` uses PASO's drug-blind splits, which assign drugs to folds
randomly. A skeptic can argue: "Randomly assigned folds may place structurally similar drugs
in both train and test. In this case, drug features (which encode structural similarity) are
evaluated against test drugs that structurally resemble training drugs — a favorable setting
for fingerprints. The null might not hold when test drugs are genuinely structurally novel."

This experiment replaces random drug-blind splits with Bemis-Murcko scaffold-stratified
folds, ensuring all drugs sharing a molecular scaffold are held out together. This is the
most structurally challenging evaluation: test drugs come from scaffolds entirely absent
from training.

## Design

| Setting | Value |
|---------|-------|
| Model | Ridge(α=1.0) |
| Cell features | RNA PCA(550) + mutation PCA(200) |
| Conditions | `morgan_fp` vs `no_drug` only |
| Splits | Bemis-Murcko scaffold 5-fold (217 unique scaffolds across 233 drugs) |
| Metric | Per-drug Pearson r |

Only two conditions (Morgan FP vs no_drug) because the question is whether the split
structure matters, not whether different representations differ.

## Scaffold stratification policy

217 unique Bemis-Murcko scaffolds across 233 drugs; target: ≈ 46–47 drugs per fold.

**Fold assignment algorithm**:
1. Compute Bemis-Murcko scaffold for each drug via RDKit `MurckoDecompose`. Drugs with no
   ring system (acyclic SMILES) receive scaffold = their canonical SMILES (treated as singleton
   scaffolds — each is its own group).
2. Sort scaffold groups by size (number of drugs) descending.
3. Assign scaffold groups to folds greedily: each new group goes to the fold with the fewest
   drugs currently assigned (min-heap bin-packing). This minimizes max fold imbalance.
4. Verify: `assert max_fold_size − min_fold_size ≤ 5` and
   `assert len(train_scaffolds ∩ test_scaffolds) == 0` for every fold (hard failure if violated).

**Edge cases**:
- Drugs with identical SMILES (rare): same SMILES → same scaffold group → same fold (correct).
- Drugs with invalid SMILES: exclude before scaffold assignment; log count; require ≤ 5
  exclusions (>5 exclusions is a data quality issue requiring investigation).
- Expected singleton count: ~16 drugs (≈ 7%) have acyclic SMILES; each becomes its own
  scaffold group and is distributed evenly across folds by the bin-packing step.

## Why this is the right falsification

Morgan fingerprints encode scaffold structure. If drug features only help for drugs
structurally related to training drugs, scaffold-blind splits are exactly where we expect
to see a larger Δ. If the null holds here, it is not a split-design artifact.

## Validation checks

- Scaffold-blind `no_drug` per-drug r ≈ 0.643 ± 0.020 (not substantially lower than PASO splits 0.631)
- Morgan FP Δ ≤ +0.005 (null holds even under structural novelty)
- n_scaffolds = 217 (verified)
- Scaffold leak assertion: `assert len(train_scaffolds ∩ test_scaffolds) == 0` for every fold
  (hard failure if any scaffold appears in both train and test)

## Prerequisites

- GDSC2 omics: `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/morgan_fp.npy`
- SMILES strings for Bemis-Murcko scaffold computation (from `data/processed/drug_smiles.csv`)
- RDKit installed (`rdkit` via uv)

## Pre-flight checks

Before running, verify:
1. `data/processed/drug_smiles.csv` present; spot-check 5 SMILES parse without error in RDKit
2. Invalid SMILES count ≤ 5 (log and exclude; > 5 requires investigation)
3. `data/processed/rna.parquet` and `data/processed/mutations.parquet` shapes match `02`
4. `data/processed/morgan_fp.npy` shape: (233, 2048)
5. After scaffold assignment: verify n_scaffolds = 217 ± 2 and max|fold_size − 46| ≤ 5
6. Scaffold leak assertion passes on dry-run before fitting any model

## How to run

```bash
sbatch experiments/03_drug_feature_null/04_split_robustness/jobs/sbatch.sh
```

Expected runtime: < 15 min (CPU, Ridge only). SLURM: `--mem=16G`, no GPU, `--time=0:30:00`.

## Output

```
report/data/metrics.json   — no_drug and morgan_fp per-drug r (mean, std, fold values, Δ)
                             per_scaffold breakdown (n_drugs per fold, scaffold diversity check)
```
