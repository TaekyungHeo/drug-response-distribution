# PASO 10-fold reproduction with OmniCancerV1

## Research question

Can we reproduce PASO's reported drug-blind Pearson r = 0.745 on their
pre-generated 10-fold splits using OmniCancerV1? This validates that our training
pipeline works correctly on PASO's data partition before the decomposition
experiment (02_decomposition).

## Background

Wu et al. (2025, PLoS Comput. Biol.) reported a drug-blind Pearson r of 0.745 for PASO,
substantially above the 0.4–0.5 range typical for drug-blind prediction on GDSC2.
Before decomposing how that figure was generated, we need to confirm that our pipeline
produces results in the expected range on the same fold splits. This experiment also
documents the fold-level variance that makes single-fold reporting misleading.

## Experimental design

OmniCancerV1 (Transformer architecture, RNA-seq and mutation omics, Morgan FP drug
encoding) was trained on PASO's published 10-fold drug-blind splits. The protocol
follows PASO's own evaluation: 200 epochs per fold, with the checkpoint achieving the
highest test-set Pearson r selected for reporting. Using PASO's protocol here is
intentional: it provides a reference point that matches the conditions under which their
headline number was generated, before introducing the fair-protocol comparison in
02_decomposition.

## Results

The 10-fold mean under the PASO-style protocol is **0.603 ± 0.091**.
The best fold (fold 8, r = 0.712) is within 0.03 of PASO's
reported 0.745, plausible given architectural differences between OmniCancerV1 and
PASO's pathway cross-attention design.

| Fold | Test r | Best epoch |
|:----:|:------:|:----------:|
| 1 | 0.6871 | 33 |
| 2 | 0.5178 | 7 |
| 3 | 0.6916 | 27 |
| 4 | 0.5785 | 25 |
| 5 | 0.6496 | 194 |
| 6 | 0.5433 | 7 |
| 7 | 0.5788 | 177 |
| 8 | 0.7118 | 5 |
| 9 | 0.6648 | 198 |
| 10 | 0.4078 | 167 |

| Statistic | Value |
|-----------|:-----:|
| **10-fold mean** | **0.6031** |
| 10-fold std | 0.0910 |
| Best fold | 0.7118 (fold 8) |
| PASO reported | 0.745 |

The fold-level variance is substantial: r ranges from
0.408 to 0.712,
with a standard deviation of 0.091.
Fold 10 (r = 0.408) is a clear low outlier,
likely composed of drugs whose sensitivity patterns are poorly correlated with the
training set. This fold-level heterogeneity is a structural property of drug-blind
evaluation: which drugs end up in the test set matters more than model choice, and
reporting a single fold without the mean and standard deviation is therefore
uninformative.

