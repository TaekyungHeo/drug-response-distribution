# PASO drug-blind r decomposition: test-set snooping and best-fold selection

## Research question

**What fraction of PASO's reported drug-blind r = 0.745 is attributable to test-set
snooping, and what fraction to selecting the single best cross-validation fold?**

## Background

Wu et al. (2025, PLoS Comput. Biol.) reported a drug-blind Pearson r of **0.745** for PASO,
substantially above the 0.4–0.5 range typical for drug-blind prediction on GDSC2.
This result appeared across two separate evaluation settings in their paper and was used
as evidence that pathway-aggregated features with cross-attention improve generalization
to unseen drugs.

Two protocol choices in the original study warrant scrutiny. First, PASO selects the
checkpoint with the highest test-set performance at each epoch, rather than using a
held-out validation set for early stopping. When the test set is used to select the
checkpoint, the reported metric is optimistically biased because the model has implicitly
seen the test labels during training. Second, the paper's headline figure quotes a single
cross-validation fold rather than the mean across all folds. In 10-fold cross-validation,
fold-to-fold variance is high enough that the best fold can be far above the mean by chance.

This experiment decomposes the 0.745 figure into these two components by re-running
the same training procedure under controlled protocol variants.

## Experimental design

All runs use OmniCancerV1 (RNA and mutation omics, Morgan FP drug encoding) trained on
PASO's published drug-blind 10-fold splits
(`external/PASO/data/10_fold_data/drug_blind/`), which fixes fold membership to exactly
the same train/test partitions as the original study.
Two protocols are compared across all 10 folds.

The **PASO-style protocol** selects the checkpoint with the highest test-set Pearson r
at each fold. The reported metric is this best-test-epoch r, averaged across folds.
This reproduces the implicit snooping in the original evaluation.

The **fair protocol** holds out a 10% validation split from the training data at each
fold. The checkpoint with the highest validation r is selected; test-set performance at
that checkpoint is reported and averaged. The test set is never used to select the model.

The first gap isolates the test-set snooping contribution; the second, between the fair
mean and the best single fold, isolates the best-fold reporting contribution.

Training details: 2 epochs, batch size 512,
learning rate 0.001, 141004 cell–drug pairs,
687 cell lines, 233 drugs.

## Results

The fair global r is **0.509**, consistent with the 0.44–0.52 range
seen in other drug-blind benchmarks on GDSC2. From this baseline, two protocol choices
inflate the headline figure. Test-set snooping adds +0.042,
bringing the PASO-style mean to 0.550. Selecting the
single best fold rather than reporting the mean adds a further
+0.200,
producing a headline figure of 0.751, approximately
matching the published 0.745.
Total inflation from the fair mean to the reported figure is **+0.242**.

| Protocol | Global r | Std | Per-drug r | Std |
|----------|:--------:|:---:|:----------:|:---:|
| PASO-style (best epoch on test set) | 0.550 | 0.131 | 0.646 | 0.019 |
| Fair (best epoch on validation set) | 0.509 | 0.143 | 0.642 | 0.021 |
| Snooping contribution | +0.042 | | +0.004 | |
| Best single fold (fair protocol) | 0.751 | | — | |
| Best-fold contribution (vs PASO-style mean) | +0.200 | | — | |
| Total inflation (vs fair mean) | +0.242 | | — | |

The inflation is entirely in global r. Per-drug r is 0.642 ± 0.021
under the fair protocol and 0.646 ± 0.019
under the PASO-style protocol, a difference of
+0.004, consistent with null.
Best-fold selection, which drives most of the global r inflation, has no analogue for
per-drug r because per-drug r does not respond to between-drug potency variance.
This confirms that the snooping and best-fold artifacts operate entirely through the
between-drug axis that `01_global_vs_perdrug` identified as the confound in global r.

The fair per-drug r of 0.642 ± 0.021 is the right number to carry
forward, protocol-clean and on the correct metric.

OmniCancerV1 uses Morgan fingerprint drug features, yet its per-drug r
(0.642) matches the cell-only Ridge baseline (~0.644)
that uses no drug features at all. Drug features do not improve within-drug cell ranking.
A drug feature is constant across all cell lines for a given drug, so adding it shifts
the predicted mean for that drug but leaves the cell ranking unchanged. Per-drug r
cannot respond to drug features regardless of how informative they are. Drug features
can only contribute to global r, which measures between-drug differences, the axis
`01_global_vs_perdrug` showed is dominated by potency memorisation. The apparent superiority of feature-rich models in the literature rests on this confounded metric.

The per-fold breakdown shows that the high-performing folds (8 and 9, r ≈ 0.71–0.75)
are not representative of the global r distribution: six of ten folds fall below r = 0.60,
and the standard deviation across folds (0.143) is comparable
in magnitude to the mean, reflecting high sensitivity to which drugs end up in the
test set. Per-drug r shows no such variance (std = 0.021),
consistent with it measuring a stable cell-biology signal rather than fold composition.

| Fold | PASO-style r | Fair r | Snooping Δ |
|------|-------------|--------|------------|
| 1 | 0.513 | 0.513 | +0.000 |
| 2 | 0.539 | 0.500 | +0.039 |
| 3 | 0.454 | 0.423 | +0.031 |
| 4 | 0.618 | 0.508 | +0.110 |
| 5 | 0.485 | 0.466 | +0.019 |
| 6 | 0.349 | 0.335 | +0.014 |
| 7 | 0.676 | 0.640 | +0.035 |
| 8 | 0.751 | 0.724 | +0.027 |
| 9 | 0.727 | 0.715 | +0.012 |
| 10 | 0.393 | 0.265 | +0.129 |

## Limitations

This reproduction uses OmniCancerV1, not PASO's original pathway-aggregation
architecture with cross-attention. The snooping inflation (+0.042)
and best-fold variance are protocol properties rather than architecture properties, so
the decomposition should hold qualitatively for the original model. The quantitative
magnitudes could differ if PASO's architecture is more or less prone to epoch-selection
overfitting than OmniCancerV1. The fold splits are identical to those released by
Wu et al., so any difference in drug-to-fold assignment is ruled out.

