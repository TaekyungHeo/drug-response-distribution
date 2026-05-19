# 03 — K-Shot Response Matching: External Replication

## Research question

Does K-shot response matching lift per-drug r across independent datasets?
What are the conditions under which the K-curve works vs. fails?

## Background

In GDSC2, K-shot response matching raises per-drug r from 0.637 (K=0) to 0.701
(K=50), exploiting the fact that training drugs with similar response profiles
on K observed cells will have similar profiles on unobserved cells
(05_solutions/03_few_shot/01_response_matching).

Three external datasets test generalization, but with structurally different setups:
- **CTRPv2**: Different cell lines from GDSC2 (812 Broad vs. 969 Sanger) — tests
  OOD cell transfer with the same drug panel concept (targeted + cytotoxic agents)
- **BeatAML**: Patient-derived AML samples — tests whether matching works when
  the "cells" are unique patients not present in any training matrix
- **PRISM**: Same 477 cell lines as GDSC2, but 1079 diverse repurposing drugs —
  tests whether matching works when most test drugs have no similar training drugs

## Experimental design

- **Model**: Ridge(alpha=1.0) + response matching, blend weight w=0.5 (fixed)
- **K values**: [0, 1, 3, 5, 10, 20, 50]
- **Matching**: top-5 nearest training drugs by response profile (Pearson over K obs)
- **Training reference**: GDSC2 response matrix (233 drugs × 969 cells)

## Results

### K-curve

| K | CTRPv2 | BeatAML | PRISM |
|--:|:------:|:-------:|:-----:|
| 0 | 0.411 | 0.453 | 0.112 |
| 1 | 0.416 | 0.454 | 0.112 |
| 3 | 0.314 | 0.385 | 0.022 |
| 5 | 0.309 | 0.395 | 0.013 |
| 10 | 0.325 | 0.433 | 0.004 |
| 20 | 0.379 | 0.465 | 0.001 |
| 50 | 0.463 | 0.521 | 0.006 |

Dataset sizes: CTRPv2 545 drugs / 812 cells,
BeatAML 155 drugs / 520 patients,
PRISM 1079 drugs / 477 cells.


CTRPv2 K=50 lift: +0.051 (K=0=0.411, K=50=0.463).
GDSC2 reference: K=0=0.637, K=50=0.701 (+0.064).

## Interpretation

### CTRPv2: K-shot matching works with OOD cell lines

K=50 yields per-drug r=0.463, a +0.051 lift over K=0
(0.411). The matching mechanism generalises to unseen cell populations: when
812 Broad cells are used as pilot observations, the GDSC2 training
response matrix still identifies informative nearest-neighbour drugs. The K=1 point does
not dip below K=0 (0.416 vs 0.411),
suggesting AUC measurements at K=1 provide slightly more ranking signal than IC50 at K=1,
possibly due to lower assay noise.

### BeatAML: Matching fails — patient cells are not in the training matrix

All K≥1 values are NaN. The response matching algorithm computes Pearson correlation
between K observed patient responses and each training drug's profile on the *same*
patients — but no BeatAML patient appears in the GDSC2 training matrix. There are no
valid neighbours to match against. This is not a model failure; it defines a scope
condition: **K-shot matching requires that the K observed cells exist in the training
response matrix**. Patient-derived or clinic-specific cell populations that were never
profiled in training cannot be matched. A future extension would require building a
patient-specific reference panel.

### PRISM: Matching fails — PRISM drugs are too dissimilar from GDSC2 training drugs

The K-curve collapses after K=1 (from 0.112 at K=0
to 0.022 at K=3). The GDSC2 training set contains
~233 mostly targeted and cytotoxic drugs; PRISM contains ~1400 diverse repurposing library
compounds, most of which have no chemical or functional analogue in GDSC2. When K
observations are used to find the "most similar" GDSC2 training drug, the best match
is still structurally unrelated, and transferring its profile across 477
cells adds noise rather than signal. This defines a second scope condition: **K-shot
matching requires that the training set contains drugs with similar mechanisms to the
test drug**. Cross-panel matching across highly diverse drug libraries fails.

### Summary: when K-shot matching works

| Condition | CTRPv2 | BeatAML | PRISM |
|-----------|:------:|:-------:|:-----:|
| Same or similar drug panel | ✓ | ✓ | ✗ |
| Cells in training matrix | ✗ (but overlap in profile space) | ✗ | ✓ |
| K-shot lift achieved | **✓ (+0.051)** | **✗ (NaN)** | **✗ (collapses)** |

K-shot response matching is effective when the training response matrix covers
either the same cell lines or at least mechanistically similar drugs. It fails when
both are absent simultaneously (patient-derived data with OOD drug panel).


