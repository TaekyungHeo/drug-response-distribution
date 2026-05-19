# 02 — Within-MoA biological ceiling

## Research question

What is the within-MoA biological ceiling for per-drug r? Do drugs sharing a
MoA class have correlated response profiles across cell lines, and by how much
does this exceed the random (cross-MoA) baseline?

## Background

Within-MoA training can only improve per-drug r if drugs in the same class have
correlated response profiles. If two drugs share a mechanism but their cell-line
rankings are uncorrelated, no training distribution change can help — the
biological signal is not there. This experiment establishes the theoretical
ceiling per MoA class before running any training interventions: classes above
the random baseline are candidates for improvement; classes at or below it are
not.

## Experimental design

- **Metric**: pairwise Pearson r between response profiles (IC₅₀ across shared cell lines) for drug pairs within each MoA class
- **Random baseline**: pairwise r for random drug pairs regardless of MoA (2353 pairs)
- **Inclusion criterion**: MoA groups with ≥3 drugs and ≥1 valid pair

## Results

**Random baseline**: mean r = 0.423
(std = 0.151, n = 2353 pairs)

### Within-MoA concordance (sorted by mean r descending)

| MoA | mean r | std r | min r | max r | n_drugs | n_pairs |
|-----|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|
| Mitosis | 0.713 | 0.090 | 0.595 | 0.862 | 7 | 21 |
| EGFR signaling | 0.697 | 0.051 | 0.598 | 0.771 | 7 | 21 |
| Chromatin other | 0.667 | 0.111 | 0.474 | 0.916 | 10 | 45 |
| Hormone-related | 0.629 | 0.056 | 0.513 | 0.696 | 4 | 6 |
| Cell cycle | 0.612 | 0.102 | 0.459 | 0.929 | 13 | 78 |
| Chromatin histone methylation | 0.594 | 0.086 | 0.422 | 0.794 | 11 | 55 |
| DNA replication | 0.579 | 0.150 | 0.163 | 0.907 | 20 | 190 |
| Genome integrity | 0.570 | 0.083 | 0.411 | 0.790 | 13 | 78 |
| ERK MAPK signaling | 0.565 | 0.231 | 0.070 | 0.950 | 13 | 78 |
| p53 pathway | 0.548 | 0.067 | 0.457 | 0.629 | 4 | 6 |
| IGF1R signaling | 0.548 | 0.126 | 0.333 | 0.714 | 6 | 15 |
| Other | 0.531 | 0.119 | 0.144 | 0.874 | 28 | 357 |
| PI3K/MTOR signaling | 0.489 | 0.147 | 0.173 | 0.869 | 26 | 325 |
| Other, kinases | 0.474 | 0.133 | 0.086 | 0.813 | 22 | 231 |
| Chromatin histone acetylation | 0.466 | 0.151 | 0.101 | 0.775 | 10 | 45 |
| RTK signaling | 0.454 | 0.142 | -0.075 | 0.724 | 12 | 66 |
| Metabolism | 0.440 | 0.108 | 0.235 | 0.604 | 6 | 15 |
| Unclassified | 0.436 | 0.169 | -0.108 | 0.919 | 34 | 561 |
| WNT signaling | 0.434 | 0.115 | 0.274 | 0.830 | 9 | 36 |
| Protein stability and degradation | 0.434 | 0.143 | 0.165 | 0.778 | 9 | 36 |
| Cytoskeleton | 0.405 | 0.102 | 0.217 | 0.550 | 5 | 10 |
| Apoptosis regulation | 0.331 | 0.164 | -0.041 | 0.846 | 13 | 78 |

## Interpretation

**20/22** MoA groups have mean r above the random baseline
(0.423). Classes substantially above the random
baseline have biological signal that MoA-stratified training can exploit.
Classes at or below random gain nothing from within-MoA training — the ceiling
is not training-distribution-limited but biologically limited.


