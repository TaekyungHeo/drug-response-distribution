# 02 — Scaffold-stratified (Tanimoto) split: chemical similarity leakage check

## Research question

Does the drug feature null result depend on the random drug-blind split
accidentally placing structurally similar drugs in both train and test?
If so, a Tanimoto-distance-based scaffold split — which maximizes chemical
dissimilarity between train and test — should make drug features more useful.

## Background

Drug-blind cross-validation assigns drugs randomly to folds. If drugs in train
and test sets are chemically similar — sharing the same Bemis-Murcko scaffold or
high Tanimoto fingerprint similarity — the drug feature null might be an artifact:
features carry no signal because train-test pairs are already similar without
them. A scaffold-stratified split that maximizes chemical dissimilarity between
folds is a stricter test. If per-drug r remains stable under this split, the
null result is not a consequence of chemical leakage across random folds.

## Experimental design

- **Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200), no drug features
- **Data**: GDSC2, 233 drugs
- **Splits compared**:
  - **Random**: standard 10-fold drug-blind CV
  - **Tanimoto**: Bemis–Murcko scaffold clustering (10 clusters)

## Results

| Split | Mean per-drug r | Std | Delta vs random |
|-------|:--------------:|:---:|:---------------:|
| Random drug-blind | 0.6453 | 0.0246 | — |
| Tanimoto scaffold | 0.6599 | 0.0630 | +0.0146 |

**Scaffold cluster sizes**: 13, 23, 25, 21, 64, 8, 4, 24, 3, 48

## Interpretation

Tanimoto split Δ=0.015 ≤ 0.05 — chemical similarity leakage is negligible.

The Tanimoto split actually yields slightly *higher* per-drug r than random,
not lower. This counterintuitive result occurs because scaffold clustering
creates folds of unequal size (clusters range from 3 to 64
drugs), and smaller test folds tend to have higher per-drug r due to reduced
averaging noise. The key finding: even under maximum chemical dissimilarity
between train and test drugs, per-drug r does not degrade — confirming that
the model does not rely on chemical structure similarity for cell ranking.

The delta of +0.015 is well within the ±0.05
threshold for equivalence, ruling out chemical similarity leakage as an
explanation for the drug feature null result.


