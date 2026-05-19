# 02 — Representation Ablation

## Research question

Does any drug representation class — structural, functional, mechanistic, or pharmacological
— improve within-drug cell-line ranking (per-drug r) in the drug-blind setting?

## Background

The hypothesis that drug features improve drug response prediction is foundational in the
field: models like PASO, DeepCDR, and DrugCell all incorporate drug representations
(fingerprints, graph neural networks, or target profiles) alongside cell features, and their
improvement over cell-only baselines is cited as evidence that chemical information helps.
However, those comparisons use global r as the primary metric, which is dominated by
between-drug potency differences that a model can "predict" without using drug features at all
(see 01_metric_decomposition).

Per-drug r isolates the within-drug ranking question: for a fixed drug, does the model
correctly rank cell lines by sensitivity? This metric cannot respond to drug-specific
constants — a drug feature shifts the predicted mean for a drug but leaves cell rankings
unchanged, unless the model uses drug-cell interactions. The drug feature null hypothesis is
therefore testable only under per-drug r: if drug features do not improve within-drug ranking,
they are not contributing the kind of information that would generalize to new drugs.

This is the headline experiment. All representation types — structural (Morgan FP),
language-model (ChemBERTa), mechanistic (ChEMBL targets), and pharmacological (LINCS) —
are evaluated under an identical protocol so that differences in outcome cannot be attributed
to protocol differences. Degenerate controls (shuffled fingerprints, random vectors) quantify
the baseline delta from adding extra dimensions to Ridge.

## Experimental design

- **Model**: Ridge regression (alpha=1.0)
- **Cell features**: RNA PCA(550) + mutation PCA(200)
- **Splits**: PASO 10-fold drug-blind CV (233 drugs, 687 cell lines)
- **Primary metric**: per-drug Pearson r (unweighted mean, drugs with >=50 cell lines)
- **Decision gate**: delta > 0.01 over no_drug baseline (Holm-Bonferroni corrected)

## Results

### Primary results table

| Condition | Per-drug r | delta vs no_drug | Holm p |
|-----------|:---:|:---:|:---:|
| no_drug (baseline) | 0.645 ± 0.025 | — | — |
| morgan_fp | 0.646 ± 0.024 | +0.001 | 0.0049 |
| chemberta (PCA-64) | 0.646 ± 0.024 | +0.001 | 0.0094 |
| chembl_targets | 0.646 ± 0.024 | +0.001 | 0.0094 |
| lincs (PCA-64, 104 drugs) | 0.665 ± 0.038 | +0.001 | 0.0094 |
| all_concat | 0.646 ± 0.024 | +0.001 | 0.0049 |
| morgan_fp_shuffled (degenerate) | 0.646 ± 0.024 | +0.001 | — |
| random_continuous (degenerate) | 0.646 ± 0.024 | +0.001 | — |

**Decision gate: delta > 0.01. No representation crosses the gate.**

### Degenerate baseline check

Shuffled Morgan FP and random continuous vectors produce delta = +0.001
and +0.001 respectively — identical to real Morgan FP
(+0.001). Adding ANY extra features to Ridge produces the same tiny
positive delta due to the mechanical effect of additional input dimensions. The "real" drug
feature signal is indistinguishable from noise.

### Alpha sensitivity

| 0.01 | 0.646 |
| 0.1 | 0.646 |
| 1.0 | 0.646 |
| 10.0 | 0.646 |
| 100.0 | 0.646 |

Per-drug r is invariant to alpha across 4 orders of magnitude, ruling out the hypothesis
that alpha=1.0 suppresses drug feature coefficients.

### Drug feature scaling

| 0.1x | +0.001 |
| 0.3x | +0.001 |
| 1.0x | +0.001 |
| 3.0x | +0.001 |
| 10.0x | +0.001 |

Scaling drug features by 10x has no effect.

### Similarity-stratified delta

| Tanimoto bin | n_drugs | delta |
|:---:|:---:|:---:|
| Low (0.0-0.3) | 154 | +0.001 |
| Mid (0.3-0.5) | 40 | +0.001 |
| High (0.5-1.01) | 39 | +0.001 |

Even for the most structurally novel drugs, Morgan FP provides no meaningful benefit.

## Interpretation

1. **No drug representation crosses the decision gate (delta > 0.01).** Structural
   (Morgan FP), learned (ChemBERTa), mechanistic (ChEMBL targets), and pharmacological
   (LINCS) representations all fail.

2. **The delta is indistinguishable from noise.** Real Morgan FP delta
   (+0.001) equals shuffled (+0.001)
   and random (+0.001). Drug features are not being used
   for drug-specific prediction.

3. **Power caveat**: MDE ~0.030 at 80% power
   (Monte Carlo gate power = 0.049). The degenerate
   baseline equivalence provides stronger evidence than the gate alone.


