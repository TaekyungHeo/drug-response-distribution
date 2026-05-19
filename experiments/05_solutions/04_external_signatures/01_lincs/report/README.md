# 01 — LINCS L1000 signatures as drug features

## Research question

Do LINCS L1000 transcriptional signatures improve global r and/or per-drug r
beyond the cell-only baseline? And do the two metrics respond differently?

## Background

All prior drug feature experiments (03_drug_feature_null) used structural
features (Morgan fingerprints, ChemBERTa) that encode what a drug looks like
chemically. LINCS L1000 is categorically different: it measures the actual
cellular transcriptional response to each drug — the functional consequence of
the drug's mechanism. If any drug feature can improve per-drug r, LINCS is the
strongest candidate. Critically, LINCS is a drug-level constant (the same
signature regardless of which cell is treated), so it can encode between-drug
potency differences (global r) but cannot differentiate within-drug cell
rankings (per-drug r). Testing both metrics dissects which axis LINCS addresses.

## Experimental design

- **Drugs**: 104 with LINCS coverage out of 233 PASO drugs
- **Drug features**: LINCS L1000 consensus signatures, LINCS PCA(64) capturing 98.0% of variance
- **Cell features**: RNA PCA(550) + mutation PCA(200)
- **Model**: Ridge(alpha=1.0), PASO 10-fold drug-blind CV
- **Control**: random vector drug features (same dimensionality as LINCS PCA)
- **Evaluation**: all conditions evaluated on the same 104-drug LINCS-covered subset

## Results

| Condition | global r | per-drug r | delta per-drug r |
|-----------|:--------:|:----------:|:----------------:|
| no_drug | 0.3007 | 0.6592 | — |
| lincs | 0.2426 | 0.6605 | 0.0012 |
| random_vector | 0.2740 | 0.6601 | 0.0009 |

Global r delta (lincs vs no_drug): -0.0580.
Per-drug r delta: 0.0012.
Random control per-drug r delta: 0.0009.

## Interpretation

LINCS signatures are a drug-level constant across cells. In principle, they
could encode between-drug potency and mechanism — but empirically, global r
*decreases* by -0.058 on the 104-drug LINCS-covered subset. The most likely
explanation is that LINCS PCA dimensions introduce noise that disrupts Ridge's
existing between-drug signal without adding compensating information at this
sample size. Per-drug r is unchanged (+0.001, matching the random vector
control at +0.001), confirming LINCS cannot inform within-drug cell ranking.
The dissociation is asymmetric: LINCS helps neither global r nor per-drug r,
but for different reasons — noise contamination on the global axis, and
structural inability to differentiate cells on the per-drug axis.


