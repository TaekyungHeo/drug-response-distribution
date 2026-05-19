# 03 — Ranking Loss: Is the Ceiling MSE-Loss-Specific?

## Research question

Ridge is trained with MSE loss but evaluated with per-drug r (a ranking metric).
Does drug-standardized training (within-drug z-scores, making MSE equivalent to
maximizing within-drug correlation) improve per-drug r?

## Background

Ridge minimizes MSE on raw IC₅₀ values. A single linear function of cell
features predicts the same ranking for every drug — the global cell-sensitivity
ordering. MSE encourages this to match the overall IC₅₀ scale, but the
per-drug r evaluation cares only about within-drug rank order. Drug-standardized
targets (z-scores computed within each drug) remove between-drug IC₅₀ scale
differences and make MSE equivalent to directly maximizing within-drug
correlation. If the per-drug r ceiling is an MSE artifact, drug-standardized
training should push it higher. If the ceiling holds, it reflects a fundamental
limit on what cell features can recover about within-drug cell ranking.

## Experimental design

- Ridge(alpha=1.0), RNA PCA(550) + mut PCA(200), PASO 10-fold drug-blind CV
- `ridge_mse`: raw ln_ic50 targets (reference)
- `ridge_rank`: drug-standardized targets (within-drug z-scores, alpha=1.0)
- `ridge_rank_01`: drug-standardized, alpha=0.1
- `ridge_rank_10`: drug-standardized, alpha=10.0

## Results

| Condition | Per-drug r | delta vs MSE |
|-----------|:---:|:---:|
| ridge_mse (reference) | 0.645 ± 0.025 | — |
| ridge_rank (alpha=1.0) | 0.648 ± 0.024 | +0.003 |
| ridge_rank (alpha=0.1) | 0.648 ± 0.024 | +0.003 |
| ridge_rank (alpha=10.0) | 0.648 ± 0.024 | +0.003 |

**The ceiling is not MSE-specific.** Drug-standardized training produces
delta = +0.003, consistent with null.
Alpha sweep (0.1 to 10.0) produces identical results.

## Interpretation

A no-drug-feature Ridge learns one linear function of cell features shared across all
drugs. Within-drug ranking of this function is the same regardless of target
standardization, because rankings are ordinal and cell features are drug-independent.
The delta of +0.003 confirms this
mechanistic reasoning: the per-drug r ceiling is not an artifact of MSE loss.

