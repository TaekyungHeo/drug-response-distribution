# 04 — Split Robustness: Scaffold-Stratified Evaluation

## Research question

Does the drug-feature null depend on the random drug-blind split accidentally placing
structurally similar drugs in both train and test? If test drugs are structurally novel,
does Morgan FP finally provide a signal for knowledge transfer?

## Background

`02_representation_ablation` uses PASO's drug-blind splits, which assign drugs to folds
randomly. A skeptic can argue that randomly constructed folds may happen to place
structurally similar drugs on both sides of the train/test boundary. In this case, drug
features that encode structural similarity are evaluated against test drugs that closely
resemble training drugs — a favorable setting where copying the training drug's profile
would work well. The null result might not hold when test drugs are genuinely
structurally novel.

Bemis–Murcko scaffold-stratified splitting addresses this: all drugs sharing a molecular
scaffold are assigned to the same fold, so the test set contains scaffolds entirely absent
from training. This is the most structurally challenging evaluation setting for drug
features. If Morgan FP can ever help, it should help here.

## Experimental design

- **Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200)
- **Data**: GDSC2 (233 drugs, 687 cell lines)
- **Splits**: Bemis–Murcko scaffold 5-fold CV — 219 unique scaffolds,
  all drugs sharing a scaffold held out together
- **Fold sizes**: 47, 46, 46, 46, 46 drugs
- **Conditions**: no drug features vs Morgan FP (2048-bit) only (the question is about
  split structure, not representation diversity)
- **Scaffold leak check**: verified no scaffold appears in both train and test

## Results

| Condition | Mean per-drug r | Std | Delta |
|-----------|:--------------:|:---:|:-----:|
| No drug features | 0.6404 | 0.0107 | — |
| Morgan FP | 0.6409 | 0.0106 | +0.0006 |

**Scaffold leak assertion**: PASSED

Morgan FP delta = +0.0006 under scaffold-stratified
splits, indistinguishable from zero and consistent with the random-split result (+0.001
in experiment 02).

## Interpretation

Even when test drugs come from scaffolds entirely absent from training — the most demanding
structural novelty setting for drug feature transfer — Morgan fingerprints provide no
benefit to within-drug cell ranking. The null is not an artifact of chemical similarity
leakage across folds.

The result has a straightforward mechanistic explanation: drug features can only help cell
ranking if the model learns scaffold-specific gene expression responses. The data contain
233 drugs across 219 scaffolds, so each scaffold has very few training
examples. The model cannot learn reliable scaffold-to-cell-ranking mappings, regardless of
whether the test scaffolds are novel or familiar.


