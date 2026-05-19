# 02 — Lineage Stratification and LINCS Subset Analysis

## Why This Experiment

Two fatal reviewer objections to the paper's central claims required direct testing:

**Objection 1 (Wessels/Hempel)**: Per-drug r = 0.65 is pan-cancer. If the model learns "cancer lineage → average sensitivity" rather than genuine within-lineage cell ranking, drug features would trivially be irrelevant (lineage classification doesn't benefit from chemical structure). The true per-drug r within each lineage could be much lower.

**Objection 2 (Lim/Collins)**: The 104 LINCS-covered drugs are well-studied targeted agents, potentially more predictable than the 129 non-covered drugs. If their baseline per-drug r (no drug features) is already high, any apparent global r benefit attributed to LINCS signatures would be inflated by subset composition, not genuine LINCS contribution.

## Method

Ridge regression, no drug features, 10-fold drug-blind CV (PASO splits).
Cell features: RNA + mutations → PCA(550 + 200 = 750 dims) fitted on unique training cells, capturing ~100% of cell variance (rank ≤ 687 unique cells < 19,193 RNA features).
Per-lineage r: restrict test predictions to cell lines from each lineage; require ≥ 5 cell-drug pairs per drug.

## Key Results

**Lineage-stratified per-drug r:**

| Lineage | Per-drug r | N cell-drug pairs |
|---------|:----------:|:-----------------:|
| Pan-cancer (overall) | 0.645 | — |
| CNS | 0.643 | 749 |
| Other | 0.595 | 6048 |
| Lung | 0.598 | 2687 |
| Breast | 0.571 | 1022 |
| Colorectal | 0.554 | 936 |
| Hematologic | 0.524 | 2267 |
| Skin | 0.484 | 711 |

**LINCS subset baseline per-drug r:**

| Subset | Per-drug r |
|--------|:----------:|
| LINCS-covered (104 drugs) | 0.665 |
| LINCS-uncovered drugs | 0.628 |

## Findings

**Objection 1 resolved**: All lineages achieve per-drug r ≥ 0.48. No lineage collapses to ~0.3, which would indicate pure lineage-composition inflation. The pan-cancer ceiling is somewhat elevated (0.645 vs worst lineage 0.484) by cross-lineage variance, but genuine within-lineage cell ranking occurs at all tested cancer types. The model is not doing lineage classification — it is ranking cells within lineages.

**Objection 2 resolved**: LINCS-covered drugs (r=0.665) and non-covered drugs (r=0.628) achieve similar baseline per-drug r (Δ = 0.037, below the 0.05 bias threshold). LINCS-covered drugs are not substantially easier to predict without drug features. This rules out large subset-composition inflation as an explanation for any apparent LINCS benefit. (Note: the actual LINCS experiment (05_solutions/04_external_signatures/01_lincs) finds LINCS decreases global r by −0.058 on the 104-drug subset; the composition confound is moot.)

See [report](report/README.md).
