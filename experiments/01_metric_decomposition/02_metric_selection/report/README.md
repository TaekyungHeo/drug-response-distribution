# Which metric should replace global Pearson r?

## The question

`01_global_vs_perdrug` established that global Pearson r conflates between-drug
potency ranking with within-drug cell sensitivity ranking, and that per-drug r is
the appropriate primary metric for joint models.
This experiment asks the follow-up question: among candidate per-drug metrics —
Pearson r, Spearman r, Kendall τ, NDCG@5, and per-drug R² — which one should
be used in practice?

The answer is expected to confirm Pearson r, since Spearman and Pearson are nearly
equivalent at n ≈ 22 (median test cells per drug under drug-blind split).
This experiment exists to verify that expectation rather than to discover it.

## Design

A Ridge regression model (RNA-seq only, PCA-compressed to 550 + 200 mutation
components, Morgan FP 2048-bit) is trained on a 5-fold drug-blind cross-validation
scheme.
For each (drug, fold) pair with ≥ 5 test cell lines, all five metrics are computed.
Bootstrap confidence intervals (200 resamples) are computed per (drug, fold) to
measure estimation efficiency.

**Decision rule**: ratio = median CI width(r_p) / median CI width(r_s).
If ratio > 1.1, Spearman is meaningfully more efficient and should be preferred.
If ratio ≤ 1.1, Pearson is at least as efficient and is preferred for literature
compatibility.

## Results

### Bootstrap CI widths (5-fold drug-blind, 117 drug-fold entries)

| Metric | Median CI width | IQR |
|--------|----------------|-----|
| Pearson r | **0.0899** | 0.0400 |
| Spearman r | 0.1004 | 0.0330 |
| Kendall τ | 0.0824 | 0.0142 |
| NDCG@5 | 0.1914 | 0.0950 |
| per-drug R² | 0.3748 | 0.6983 |

Pearson/Spearman CI ratio: **0.8956**
(threshold 1.1 → Pearson selected).

### Inter-metric rank correlation (Spearman, all drug-fold entries)

| | r_s | τ | NDCG@5 | R² |
|---|---|---|---|---|
| r_p | 0.9718 | 0.9739 | 0.4969 | 0.0369 |
| r_s | — | 0.9983 | 0.4789 | 0.0555 |
| τ | — | — | 0.4900 | 0.0468 |
| NDCG@5 | — | — | — | 0.1279 |

### Predictor 2A sanity check (constant drug-mean predictor, 117 drug-fold entries)

A constant predictor that outputs the per-drug mean IC₅₀ for every cell is the
null model for within-drug sensitivity ranking.
It should score 0.0 on any correlation-based metric by construction,
and a good metric must reject it.

| Metric | Predictor 2A median | Expected | Pass? |
|--------|---------------------|----------|-------|
| Pearson r | 0.0000 | 0.0000 | yes |
| Spearman r | 0.0000 | 0.0000 | yes |
| Kendall τ | 0.0000 | 0.0000 | yes |
| NDCG@5 | 0.5367 | 0.0000 | NO — tie artifact |
| per-drug R² | 0.0000 | 0.0000 | yes |

Pearson r, Spearman r, and Kendall τ all return 0.0 for a constant predictor
(zero-std prediction → correlation undefined, handled as 0.0 by convention).
Per-drug R² also returns 0.0 (SS_res = SS_tot for mean predictor).
NDCG@5 does not return 0.0 for constant predictions: scikit-learn's default
tie-handling assigns a non-zero score by averaging over all possible orderings
of tied predictions, yielding an inflated value that does not reflect prediction
quality. This artifact is a structural problem with NDCG@5 as a metric.

## Interpretation

**Pearson r is the right choice.**

The CI ratio of 0.8956 is well below the 1.1
threshold: Pearson CIs are actually *narrower* than Spearman CIs at n ≈ 22,
meaning Pearson is at least as efficient an estimator.

Pearson, Spearman, and Kendall τ are effectively interchangeable rank-wise:
Pearson vs Spearman Spearman correlation = 0.9718,
Pearson vs τ = 0.9739.
A model ranked highly by Pearson r will be ranked almost identically by Spearman r
or τ — the three metrics agree nearly perfectly on drug-by-drug rankings.

NDCG@5 and per-drug R² are poor choices.
NDCG@5 CI width (0.1914) is
2.1× wider than Pearson,
and its rank correlation with Pearson r is only 0.4969 —
it is measuring something meaningfully different and with high noise.
Per-drug R² is the worst performer: CI width 0.3748,
IQR 0.6983, rank correlation with Pearson r only
0.0369.
Per-drug R² is undefined or degenerate for short drug-blind test sets and should not
be used as a primary metric.

All subsequent experiments in this series use **per-drug Pearson r** as the primary
metric, with Spearman r and global Pearson r reported as secondary diagnostics.

## Prior work

No prior work systematically compares these five metrics at the per-drug level for
drug response prediction.
Sealfon et al.
([J. Cheminformatics 17:28, 2025](https://doi.org/10.1186/s13321-025-00965-x))
endorse per-drug Pearson r but do not compare it to Spearman, τ, NDCG, or R².
Hafner et al.
([Nat Methods 2017](https://doi.org/10.1038/nmeth.4285))
argue for area-under-curve metrics in dose-response settings, but their concern
is response curve fitting rather than model ranking — a different question.
This experiment fills the gap by providing a data-driven comparison on GDSC2
under the drug-blind protocol.

## Limitations

The comparison is conducted under a single split protocol (drug-blind) and a single
model class (Ridge regression on RNA-seq).
Results may differ under mixed-set or cell-blind splits where test set sizes per drug
are larger.
NDCG@5 behavior depends heavily on the top-k definition and tie-handling; the
implementation here uses scikit-learn's default, which treats tied predictions
near-optimally — NDCG@5 may look artificially high for constant-within-drug
predictors.
The bootstrap CI width comparison measures estimation efficiency, not statistical
power; a wider CI does not necessarily mean the metric is wrong, only that it
requires more data per drug to be reliable.

