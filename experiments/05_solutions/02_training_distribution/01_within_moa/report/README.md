# 01 — Within-MoA Leave-One-Drug-Out Ridge

## Research question

Does restricting Ridge training to same-MoA drugs improve within-MoA per-drug r
beyond the all-drug baseline (~0.644)?

## Background

Under all-drug training, Ridge learns a drug-agnostic cell ordering — a single
function of cell features averaged across all drug mechanisms. This forces the
model to capture a compromise representation that may be suboptimal for any
individual MoA. By training only on same-MoA drugs, the model can specialize:
it learns which cell-state dimensions predict sensitivity specifically to that
mechanism. The 02_moa_ceiling experiment confirmed that ERK MAPK and EGFR
signaling have high within-MoA profile concordance — making them the primary
candidates for improvement here.

## Experimental design

| Component | Description |
|-----------|-------------|
| Data | GDSC2, 226 drugs across 21 MoA classes |
| Model | Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200), cell features only |
| Procedure | Within-MoA LOO: for each MoA with ≥3 drugs, hold out one drug, train on remaining same-MoA drugs, predict held-out |
| Baseline | All-drug Ridge per-drug r from 01_diagnosis/01_moa_performance |
| Metric | Per-drug Pearson r, macro-averaged within each MoA class |

## Results

**Overall**: all-drug mean r = 0.644, within-MoA mean r = 0.664 (226 drugs)

### Per-MoA table (sorted by delta descending)

| MoA | all-drug r | within-MoA r | delta | n_drugs |
|-----|-----------|-------------|-------|---------|
| EGFR signaling | 0.425 | 0.799 | +0.375 | 7 |
| ERK MAPK signaling | 0.427 | 0.723 | +0.296 | 11 |
| PI3K/MTOR signaling | 0.610 | 0.702 | +0.091 | 23 |
| Chromatin other | 0.714 | 0.798 | +0.084 | 10 |
| IGF1R signaling | 0.608 | 0.665 | +0.057 | 5 |
| DNA replication | 0.693 | 0.735 | +0.042 | 20 |
| RTK signaling | 0.605 | 0.640 | +0.035 | 12 |
| Mitosis | 0.772 | 0.788 | +0.017 | 7 |
| Hormone-related | 0.671 | 0.687 | +0.016 | 4 |
| Chromatin histone methylation | 0.731 | 0.731 | -0.000 | 10 |
| Apoptosis regulation | 0.524 | 0.515 | -0.009 | 13 |
| Genome integrity | 0.742 | 0.730 | -0.012 | 13 |
| Cell cycle | 0.767 | 0.748 | -0.019 | 11 |
| WNT signaling | 0.615 | 0.592 | -0.023 | 9 |
| Other | 0.743 | 0.720 | -0.023 | 19 |
| Unclassified | 0.562 | 0.537 | -0.024 | 11 |
| Other, kinases | 0.648 | 0.611 | -0.036 | 15 |
| Protein stability and degradation | 0.647 | 0.582 | -0.065 | 9 |
| p53 pathway | 0.717 | 0.649 | -0.068 | 4 |
| Chromatin histone acetylation | 0.636 | 0.520 | -0.115 | 7 |
| Metabolism | 0.592 | 0.206 | -0.386 | 6 |

### Focus classes

- **EGFR signaling**: all-drug 0.425 → within-MoA 0.799 (delta +0.375, n=7)
- **ERK MAPK signaling**: all-drug 0.427 → within-MoA 0.723 (delta +0.296, n=11)
- **Mitosis**: all-drug 0.772 → within-MoA 0.788 (delta +0.017, n=7)
- **Cell cycle**: all-drug 0.767 → within-MoA 0.748 (delta -0.019, n=11)

## Interpretation

If within-MoA training substantially exceeds 0.644 for ERK MAPK / EGFR
signaling, training distribution is the first known mechanism for exceeding the
cell-mean oracle. If easy classes (Mitosis, Cell cycle) also gain, the effect
may be an artifact of reduced training set size rather than mechanism-specific
learning. The 03_onehot_control experiment dissociates training distribution
from representation to confirm the mechanism.


