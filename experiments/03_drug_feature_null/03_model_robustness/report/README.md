# 03 — Model Robustness: Is the Null Ridge-Specific?

## Research question

Does the drug-feature null hold under a nonlinear model (OmniCancerV1 Transformer)?
If Morgan FP helps under a Transformer but not Ridge, the null is a Ridge artifact
rather than a fundamental property of the drug-blind prediction setting.

## Background

`02_representation_ablation` establishes the null using Ridge regression throughout. Ridge
is a linear model: it cannot learn interactions between drug features and cell features that
are non-additive. A skeptic can reasonably argue that a nonlinear model with sufficient
capacity might extract drug-feature value that Ridge ignores — for example, by learning
which genes are relevant only for certain drug classes, or by modeling drug-cell feature
interactions that a linear kernel cannot represent.

OmniCancerV1 is a Transformer encoder with 4 layers, 256-dim, and 8 attention heads. It
takes RNA, mutation, and drug features as separate input tokens, and uses self-attention
across these tokens at each layer. This architecture can, in principle, learn arbitrary
drug-cell interactions — if Morgan FP carries any extractable signal for within-drug cell
ranking, OmniCancerV1 should find it where Ridge cannot.

## Experimental design

- **Model**: OmniCancerV1 (Transformer encoder, 4 layers, 256-dim, 8 heads)
- **Conditions**: `morgan_fp` (2048-bit) vs `no_drug` (zero vector)
- **Cell features**: RNA + mutations (all features, no PCA — Transformer handles dimensionality)
- **Splits**: PASO 10-fold drug-blind CV (233 drugs, 687 cell lines)
- **Epochs**: 200 per fold
- **Checkpoint selection**: best validation per-drug r (10% drug-blind val holdout per fold)
- **Primary metric**: per-drug Pearson r
- **Decision gate**: Δ > 0.01 over `no_drug` baseline (same gate as experiment 02)


## Results

| Fold | morgan_fp | no_drug | delta |
|:----:|:---------:|:-------:|:-----:|
| 0 | 0.633 | 0.600 | +0.032 |
| 1 | 0.628 | 0.613 | +0.016 |
| 2 | 0.676 | 0.650 | +0.026 |
| 3 | 0.630 | 0.637 | -0.007 |
| 4 | 0.674 | 0.661 | +0.013 |
| 5 | 0.637 | 0.649 | -0.011 |
| 6 | 0.628 | 0.625 | +0.003 |
| 7 | 0.699 | 0.669 | +0.030 |
| 8 | 0.662 | 0.670 | -0.008 |
| 9 | 0.656 | 0.674 | -0.018 |

| Summary | Value |
|---------|:-----:|
| **morgan_fp mean** | **0.6524** |
| no_drug mean | 0.6448 |
| **10-fold mean delta** | **+0.0076** |
| Decision gate | Δ > 0.01 |
| Gate crossed? | **NO** |

The 10-fold mean delta is **+0.0076**, below the Δ > 0.01
decision gate. **The null holds under the Transformer.**

Fold-level deltas are highly variable: 10 folds range from
+0.0323
to positive values exceeding +0.03, with both positive and negative deltas across folds.
This fold-level variance makes any single-fold comparison unreliable; the 10-fold mean
is the appropriate summary.

## Interpretation

The Transformer's null delta of +0.0076 is somewhat larger than
Ridge's (≈ +0.001 in experiment 02) but still below the decision gate. The difference from
Ridge could reflect (a) the Transformer's larger capacity learning weak drug-cell interaction
signals, (b) fold-level variance, or (c) the lack of degenerate controls (no shuffled
fingerprint condition was run). Without a degenerate control, we cannot rule out that the
+0.0076 delta is a dimensionality artifact rather than drug-content
signal — the same artifact that explained Ridge's +0.001 entirely.

Two key observations constrain interpretation:

1. **Fold-level variance dominates.** The delta swings from negative to > +0.03 across folds,
   indicating high sensitivity to which drugs end up in each test set. Mean delta < 0.01
   is not driven by a consistent positive signal but by averaging across mixed-sign folds.

2. **Absolute per-drug r is consistent with Ridge.** Both `morgan_fp` and `no_drug` achieve
   per-drug r ≈ 0.64–0.65, matching
   Ridge's ~0.645. The Transformer's additional capacity does not improve the baseline, let
   alone the drug-feature contribution.

## Limitations

No degenerate control (shuffled fingerprints or random vectors) was run under the Transformer.
In experiment 02, real Morgan FP delta = shuffled delta, confirming the Ridge delta was a
dimensionality artifact. Without the analogous check here, the Transformer delta of
+0.0076 cannot be fully attributed to drug-content signal. A
follow-up with a shuffled control would resolve this ambiguity — if shuffled Morgan FP also
produces Δ ≈ +0.0076, the entire delta is dimensional noise.

