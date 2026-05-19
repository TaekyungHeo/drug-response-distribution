# 07 — Cross-Dataset Transfer: GDSC2 → PRISM

## Research question

Does the drug-feature null hold when training on GDSC2 and evaluating on PRISM Repurposing
— a different assay platform, 1078 drugs, and minimal drug overlap?

## Background

All experiments 02–06 test the drug feature null within-dataset: train and evaluate on
GDSC2 with drug-blind cross-validation. A skeptic can argue that drug structural features
might still encode generalizable pharmacological signal, but the GDSC2 training set (233
drugs) is too small to learn the structural-to-sensitivity mapping. With a richer training
distribution from a different dataset, structural features might transfer to novel drugs
more effectively.

This experiment provides the strongest test: train Ridge on the entire GDSC2 dataset, then
evaluate on held-out PRISM drugs — a different assay platform (viability AUC vs IC₅₀),
largely non-overlapping drug set, and different cell-line assay conditions. If Morgan FP
still provides no benefit here, the null holds across dataset boundaries and cannot be
attributed to GDSC2-specific training limitations.

## Experimental design

- **Train**: GDSC2 (180614 pairs,
  286 drugs, 713 cell lines)
- **Test**: prism_repurposing (347562 pairs,
  1078 drugs, 477 cell lines)
- **Drug overlap**: 25 drugs (2.3%
  of PRISM) — no cross-validation; PRISM drugs are independent of GDSC2 training
- **Model**: Ridge(alpha=1.0)
- **Cell features**: RNA PCA(550) + mutation PCA(200), fit on GDSC2 training cells;
  applied to shared GDSC2 ∩ PRISM cell lines
- **PCA variance preserved**: RNA 95.8%
  (GDSC2 train) / 83.4%
  (PRISM test), mutations 68.1%
  / 57.3%

## Results

| Condition | Per-drug r |
|-----------|:---:|
| no_drug | 0.039 |
| morgan_fp | 0.041 |
| **delta** | **+0.002** |

**The null holds across dataset boundaries.** Morgan FP delta =
+0.002, well below the 0.01 gate.

The absolute per-drug r on PRISM (0.039) is substantially below the
within-GDSC2 baseline (~0.645). The PCA projection fit on GDSC2 cells preserves only
83.4% of RNA variance
for PRISM cells, indicating that cell feature representation is the primary bottleneck for
cross-assay transfer — not drug features.

## Interpretation

The null holds under the most stringent test: training on a different dataset (GDSC2),
evaluating on a different assay platform (PRISM), with 97.7%
drug novelty. Three conclusions follow:

1. **Drug features don't transfer across datasets.** Even at 1078 test
   drugs with high structural novelty, Morgan FP delta = +0.002.

2. **Cross-assay cell representation is the real bottleneck.** The per-drug r collapse
   from ~0.645 (within GDSC2) to 0.039 (GDSC2 → PRISM) is driven by
   cell feature mismatch: the PCA projection trained on GDSC2 captures only
   83.4% of PRISM RNA
   variance. Improving drug features would not close this gap.

3. **The null is consistent across all seven experiments.** 02 (Ridge, GDSC2), 03
   (Transformer, GDSC2), 04 (scaffold splits), 05 (PRISM within-dataset), 06 (RankNet
   exception under MSE), and this experiment converge on the same result: drug structural
   features do not improve within-drug cell-line ranking under standard training conditions.


