# 01_lincs — LINCS L1000 consensus signatures as drug features

## Research question

Do LINCS L1000 transcriptional signatures, used as drug features in Ridge,
improve global r and/or per-drug r beyond the cell-mean oracle (0.644)?

The key distinction from structural drug features (Morgan FP, descriptors) tested
in 03_drug_feature_null is that LINCS is a FUNCTIONAL measurement: what the drug
does to cells, not what the drug looks like. Structural features encode chemistry;
LINCS encodes biology. If structure fails because it cannot capture mechanism of
action, a direct measurement of mechanism might succeed.

However, the expected pattern is asymmetric: LINCS should improve global r
(between-drug scale prediction) by encoding drug potency and mechanism, but NOT
per-drug r (within-drug cell ranking), because LINCS consensus signatures are
drug-level constants — they do not vary across cell lines and therefore cannot
tell you which cells are more or less sensitive.

## Hypothesis

**Null**: LINCS signatures, like structural features, yield per-drug r delta ≈ 0
over the cell-mean oracle. Global r improvement is also negligible.

**Alternative**: LINCS improves global r substantially (+0.17) by capturing
between-drug potency/mechanism, but per-drug r delta ≈ 0 because LINCS is a
drug-level constant that cannot inform within-drug cell ranking.

Expected magnitudes (on the 104 LINCS-covered drugs):
- no_drug baseline: global r ~ 0.48, per-drug r ~ 0.64
- LINCS PCA(64): global r ~ 0.65, per-drug r ~ 0.64
- Global r delta: ~ +0.17
- Per-drug r delta: ~ 0.00

Measurement ceiling = 0.754 bounds per-drug r for all methods.

## Design

**Data**: GDSC2, 233 drugs total, ~104 drugs with LINCS L1000 consensus
signatures. 687 cell lines. PASO 10-fold drug-blind CV.

**Drug features**: LINCS L1000 consensus transcriptional signatures (978 landmark
genes), reduced to PCA(64). Signatures are per-drug constants (consensus across
cell lines and doses).

**Model**: Ridge(alpha=1.0), cell features = RNA PCA(550) + mutation PCA(200),
drug features = LINCS PCA(64). Kronecker product or concatenated interaction
features between cell and drug features.

**Procedure**:
1. Load LINCS L1000 consensus signatures for all available drugs.
2. Match LINCS drugs to GDSC2 drug IDs. Record the overlap (expected ~104/233).
3. PCA-reduce LINCS signatures to 64 dimensions.
4. Run two conditions on the SAME 104-drug subset:
   a. **no_drug**: Ridge with cell features only, trained and evaluated on the
      104 LINCS-covered drugs only.
   b. **lincs**: Ridge with cell features + LINCS PCA(64), trained and evaluated
      on the same 104 drugs.
5. For each condition, compute:
   - Global Pearson r (all predictions pooled)
   - Per-drug Pearson r (macro-averaged across drugs)
6. Use PASO 10-fold drug-blind CV throughout.

**Critical control**: Both conditions must be evaluated on the SAME 104-drug
subset. Comparing 233-drug no_drug vs 104-drug LINCS is invalid because the drug
subsets differ in difficulty.

**Metric**: Global Pearson r AND per-drug Pearson r, both on the 104-drug subset.

## Validation checks

- no_drug per-drug r on 104 drugs must be close to the full 233-drug baseline
  (~0.637). If it differs substantially, the 104-drug subset has different
  difficulty and this must be reported.
- Per-drug r must not exceed measurement ceiling (0.754).
- LINCS PCA(64) variance explained should be reported (expect >80%).
- Drug overlap count must be verified (expect ~104).
- Global r improvement should be decomposable: check whether it comes from
  better between-drug scale (mean prediction per drug) or within-drug ranking.
- Sanity check: replace LINCS features with random vectors of same dimension.
  This must yield delta ≈ 0 for both metrics.

## Output

**`report/data/results.json`** schema:
```json
{
  "drug_overlap": {
    "n_gdsc2": 233,
    "n_lincs": 978,
    "n_overlap": 104,
    "overlap_drugs": ["Trametinib", "..."]
  },
  "lincs_pca": {
    "n_components": 64,
    "variance_explained": 0.85
  },
  "comparison": {
    "no_drug": {
      "global_r": 0.48,
      "per_drug_r": 0.64,
      "n_drugs": 104
    },
    "lincs": {
      "global_r": 0.65,
      "per_drug_r": 0.64,
      "n_drugs": 104
    },
    "random_control": {
      "global_r": 0.48,
      "per_drug_r": 0.64,
      "n_drugs": 104
    }
  },
  "per_drug": [
    {
      "drug": "Trametinib",
      "drug_id": 1372,
      "no_drug_r": 0.31,
      "lincs_r": 0.32,
      "delta": 0.01
    }
  ]
}
```

**`report/data/lincs_comparison.csv`**: flat table (drug, drug_id, no_drug_global_r, lincs_global_r, no_drug_per_drug_r, lincs_per_drug_r).

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- LINCS: LINCS L1000 consensus signatures (preprocessed)
- Baseline: cell-mean oracle r = 0.644 (from 05_ceilings)
- Prior result: 03_drug_feature_null (structural features don't help per-drug r)
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`

## Resources

CPU only, <15 min, --mem=16G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/04_external_signatures/01_lincs/jobs/run.py
```

## Downstream use

If LINCS helps global r but not per-drug r, this confirms:
- Drug-level constants improve between-drug predictions but cannot break the
  cell-mean oracle for within-drug ranking.
- This is consistent with 03_drug_feature_null: structure also cannot break it.
- The distinction is global r (LINCS helps) vs per-drug r (nothing drug-level helps).

Feeds into:
- `02_lincs_prediction` (can LINCS be predicted from structure?)
- `05_combinations/02_lincs_x_moa` (LINCS + within-MoA: orthogonal axes?)
