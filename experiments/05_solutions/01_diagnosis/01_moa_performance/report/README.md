# 01 — Per-MoA Model Performance Landscape

## Research question

Which MoA classes does the all-drug Ridge model predict well, and which does it
predict poorly? And does the difficulty of a class reflect a low biological
ceiling (fundamentally hard) or a gap the training distribution could close?

## Background

All-drug training produces per-drug r ≈ 0.645 averaged across all drugs. But
this average hides large variation: some MoA classes may already be near their
ceiling while others are far below it. Before designing MoA-stratified
interventions, we need a diagnostic map — which classes are hard, which are
easy, and whether the hard ones have room for improvement. This motivates the
MoA ceiling experiment (02_moa_ceiling) and the within-MoA training experiments
that follow.

## Experimental design

| Component | Description |
|-----------|-------------|
| Data | GDSC2, 233 drugs, PASO 10-fold drug-blind CV |
| Model | Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200), cell features only |
| Metric | Per-drug Pearson r, macro-averaged within each MoA class |
| MoA source | PASO Target Pathway annotations |

## Results

**Grand mean per-drug r: 0.645** (233 drugs)

### Per-MoA table (sorted by mean r descending)

| MoA | mean r | std r | n_drugs | flag |
|-----|--------|-------|---------|------|
| Mitosis | 0.772 | 0.034 | 7 |  |
| Cell cycle | 0.767 | 0.036 | 11 |  |
| Other | 0.743 | 0.053 | 19 |  |
| Genome integrity | 0.742 | 0.034 | 13 |  |
| Chromatin histone methylation | 0.731 | 0.063 | 10 |  |
| p53 pathway | 0.717 | 0.064 | 4 |  |
| Chromatin other | 0.714 | 0.038 | 10 |  |
| DNA replication | 0.693 | 0.109 | 20 |  |
| JNK and p38 signaling | 0.676 | 0.048 | 2 | <3 drugs |
| Hormone-related | 0.671 | 0.060 | 4 |  |
| ABL signaling | 0.662 | 0.000 | 1 | <3 drugs |
| Other, kinases | 0.648 | 0.120 | 15 |  |
| Protein stability and degradation | 0.647 | 0.101 | 9 |  |
| Chromatin histone acetylation | 0.636 | 0.123 | 7 |  |
| WNT signaling | 0.615 | 0.079 | 9 |  |
| PI3K/MTOR signaling | 0.610 | 0.088 | 23 |  |
| IGF1R signaling | 0.608 | 0.061 | 5 |  |
| RTK signaling | 0.605 | 0.116 | 12 |  |
| Cytoskeleton | 0.594 | 0.057 | 2 | <3 drugs |
| Metabolism | 0.592 | 0.133 | 6 |  |
| Unclassified | 0.562 | 0.211 | 11 |  |
| Apoptosis regulation | 0.524 | 0.143 | 13 |  |
| ERK MAPK signaling | 0.427 | 0.168 | 11 |  |
| EGFR signaling | 0.425 | 0.082 | 7 |  |

### Unannotated drugs

| | mean r | n_drugs |
|--|--------|---------|
| Unannotated | 0.777 | 2 |

## Interpretation

MoA classes with high per-drug r under all-drug training are already well-served
by the baseline model. MoA classes with low per-drug r are candidates for
MoA-stratified intervention — but only if their within-MoA biological ceiling is
high (02_moa_ceiling). If a class is hard because its ceiling is low (drug
response profiles are uncorrelated within the class), no training distribution
change can help.


