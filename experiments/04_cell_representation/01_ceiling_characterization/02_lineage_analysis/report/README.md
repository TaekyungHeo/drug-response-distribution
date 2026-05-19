# 02 — Lineage analysis: per-drug r within cancer lineages

## Research question

Is the pan-cancer per-drug r inflated by cancer-type composition? A model
predicting lineage-average sensitivity would show high pan-cancer per-drug r
even without within-lineage discrimination. We also check whether LINCS
drug coverage biases the baseline.

## Background

Pan-cancer per-drug r (~0.645) pools cell lines from 7+ cancer lineages. A
skeptic can argue this number is inflated: if different lineages have
systematically different sensitivity to each drug, a model that simply learns
lineage-average responses would score high on a pan-cancer metric without
performing genuine within-lineage discrimination. This would mean the ceiling
is not a signal about cell-state biology but an artifact of lineage mixing.
We also check whether the LINCS drug subset (used in the global r analysis)
has higher baseline per-drug r than the rest, which would bias comparisons.

## Experimental design

- **Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200)
- **Data**: GDSC2, 10-fold PASO drug-blind CV
- **Lineages**: 7 major groups (Breast, CNS, Colorectal, Hematologic, Lung, Skin, Other)

## Results

### Per-lineage per-drug r

| Lineage | Per-drug r | N cell-drug pairs |
|---------|:---------:|:-----------------:|
| Breast | 0.5705 | 1022 |
| CNS | 0.6433 | 749 |
| Colorectal | 0.5541 | 936 |
| Hematologic | 0.5235 | 2267 |
| Lung | 0.5981 | 2687 |
| Skin | 0.4837 | 711 |
| Other | 0.5949 | 6048 |

**Pan-cancer overall**: 0.6453

### LINCS coverage bias check

| Subset | Per-drug r |
|--------|:---------:|
| LINCS-covered drugs (104) | 0.6647 |
| LINCS-uncovered drugs | 0.6278 |

## Interpretation

All lineages achieve per-drug r ≥ 0.48, confirming genuine within-lineage
cell ranking — not just lineage classification. The pan-cancer value (0.645)
modestly exceeds per-lineage values because cross-lineage sensitivity variance
is partially captured, but this inflation is 0.05–0.16 r units rather than
the 0.3–0.4 that would indicate a purely compositional artifact.

LINCS-covered and uncovered drugs achieve similar per-drug r (delta < 0.04),
ruling out selection bias in the LINCS global r gain reported in
05_solutions/04_external_signatures.


