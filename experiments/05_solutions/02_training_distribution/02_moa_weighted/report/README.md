# 02 — MoA-Weighted Ridge Training — Weight Sweep

## Research question

Does upweighting same-MoA training samples approach the gains of strict
within-MoA training, and what is the optimal weight W?

## Background

Strict within-MoA leave-one-out training (01_within_moa) discards all
out-of-class drugs — for small MoA classes (3–5 drugs), this severely limits
the training set. Sample weighting offers a softer alternative: keep all
training data but give higher weight to same-MoA observations. W=1 is the
all-drug baseline; increasing W progressively biases the model toward
within-MoA patterns. If performance plateaus well below the strict within-MoA
ceiling, strict exclusion is necessary. If soft weighting matches it, the
practical deployment method is simpler.

## Experimental design

| Component | Description |
|-----------|-------------|
| Data | GDSC2, PASO 10-fold drug-blind CV |
| Model | Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200), cell features only |
| Weights | Same-MoA sample weight W in [1, 2, 5, 10, 20] |
| Metric | Per-drug Pearson r, macro-averaged within each MoA class |

## Results

### Overall by weight

| W | mean per-drug r | n_drugs |
|---|-----------------|---------|
| 1 | 0.645 | 233 |
| 2 | 0.652 | 233 |
| 5 | 0.667 | 233 |
| 10 | 0.679 | 233 |
| 20 | 0.687 | 233 |

### Per-MoA breakdown

| MoA | W=1 | W=2 | W=5 | W=10 | W=20 | n_drugs |
|-----|--------|--------|--------|--------|--------|---------|
| ABL signaling | 0.662 | 0.662 | 0.662 | 0.662 | 0.662 | 1 |
| Apoptosis regulation | 0.524 | 0.529 | 0.539 | 0.546 | 0.547 | 13 |
| Cell cycle | 0.767 | 0.769 | 0.772 | 0.773 | 0.770 | 11 |
| Chromatin histone acetylation | 0.636 | 0.637 | 0.639 | 0.640 | 0.637 | 7 |
| Chromatin histone methylation | 0.731 | 0.734 | 0.740 | 0.745 | 0.749 | 10 |
| Chromatin other | 0.714 | 0.724 | 0.746 | 0.769 | 0.790 | 10 |
| Cytoskeleton | 0.594 | 0.594 | 0.596 | 0.599 | 0.604 | 2 |
| DNA replication | 0.693 | 0.707 | 0.729 | 0.740 | 0.743 | 20 |
| EGFR signaling | 0.425 | 0.444 | 0.496 | 0.566 | 0.655 | 7 |
| ERK MAPK signaling | 0.427 | 0.459 | 0.534 | 0.609 | 0.673 | 11 |
| Genome integrity | 0.742 | 0.744 | 0.747 | 0.748 | 0.747 | 13 |
| Hormone-related | 0.671 | 0.673 | 0.680 | 0.690 | 0.703 | 4 |
| IGF1R signaling | 0.608 | 0.613 | 0.626 | 0.641 | 0.660 | 5 |
| JNK and p38 signaling | 0.676 | 0.677 | 0.681 | 0.688 | 0.699 | 2 |
| Metabolism | 0.592 | 0.589 | 0.577 | 0.552 | 0.501 | 6 |
| Mitosis | 0.772 | 0.777 | 0.789 | 0.799 | 0.806 | 7 |
| Other | 0.743 | 0.745 | 0.747 | 0.746 | 0.742 | 19 |
| Other, kinases | 0.648 | 0.649 | 0.651 | 0.651 | 0.645 | 15 |
| p53 pathway | 0.717 | 0.717 | 0.719 | 0.720 | 0.719 | 4 |
| PI3K/MTOR signaling | 0.610 | 0.631 | 0.667 | 0.691 | 0.703 | 23 |
| Protein stability and degradation | 0.647 | 0.647 | 0.648 | 0.646 | 0.641 | 9 |
| RTK signaling | 0.605 | 0.612 | 0.626 | 0.638 | 0.648 | 12 |
| Unclassified | 0.562 | 0.565 | 0.573 | 0.580 | 0.584 | 11 |
| WNT signaling | 0.615 | 0.617 | 0.621 | 0.625 | 0.628 | 9 |

## Interpretation

Hard classes (ERK MAPK, EGFR) should show monotonic improvement with W as the
model is progressively biased toward within-MoA patterns. Easy classes (Mitosis,
Cell cycle) should be flat or slightly declining — over-weighting same-MoA
samples for already-easy classes does not help and reduces data efficiency.
The optimal W balances these two effects across all classes.


