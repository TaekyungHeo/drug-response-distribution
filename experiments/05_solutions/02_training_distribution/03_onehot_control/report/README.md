# 03 — MoA One-Hot as Feature: Representation Control

## Research question

Does adding MoA identity as an input feature improve per-drug r? If not, the
within-MoA training gains (01_within_moa) are a training distribution effect,
not a representation effect.

## Background

Within-MoA training substantially improves per-drug r for ERK MAPK and EGFR
signaling. One alternative explanation: the improvement comes not from the
training distribution change but from the model having access to MoA
information. If the model knew which MoA class each drug belongs to, it could
shift predictions accordingly. This experiment tests that explanation directly:
MoA one-hot encoding gives the model identical MoA information as an input
feature, but without changing the training distribution. If per-drug r does
not improve, the mechanism is definitively distribution, not representation.

## Experimental design

| Component | Description |
|-----------|-------------|
| Data | GDSC2, 233 drugs, PASO 10-fold drug-blind CV |
| Model | Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200) |
| Conditions | baseline (cell features only) vs onehot (cell + MoA one-hot) |
| MoA classes | 24 pathway categories |
| Primary metric | Per-drug Pearson r |
| Diagnostic metric | Global Pearson r |

## Results

| Condition | Per-drug r | Global r | Delta per-drug r | Delta global r |
|-----------|-----------|----------|-------------------|----------------|
| Baseline | 0.645 | 0.321 | — | — |
| + MoA one-hot | 0.645 | 0.363 | +0.000 | +0.041 |

**Conclusion**: MoA as representation does not improve per-drug r

## Interpretation

A feature that is constant for all cells treated with the same drug (MoA
one-hot) can only shift the predicted mean for that drug, not change the cell
ranking. Ridge with such a feature improves global r (by capturing drug-level
variance) but not per-drug r. This confirms that the within-MoA training gains
in 01_within_moa arise from training distribution — which drugs co-train — not
from MoA identity as a feature. The same information, routed differently,
produces opposite outcomes for per-drug r.


