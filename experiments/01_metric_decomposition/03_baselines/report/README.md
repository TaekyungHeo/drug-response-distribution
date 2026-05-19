# Cell-only prediction ceiling

## Research question

**What is the ceiling on within-drug cell sensitivity ranking (per-drug r) using cell features alone?**

## Background

The preceding experiments in this series established two things: global Pearson r is an
unreliable primary metric for drug response prediction because between-drug potency variance
dominates IC₅₀ measurements (~68%), and per-drug Pearson r is the appropriate replacement
because it measures within-drug cell ranking directly.

With the metric settled, the next question is where to set the baseline. A cell-only model
uses omics measurements of cell lines (RNA-seq, mutations, copy number, etc.) but has no
information about individual drugs. It assigns the same predicted sensitivity profile to
every drug for a given cell line. This is the natural starting point, representing what
can be learned from cell biology alone before any drug-specific information enters the
picture.

The theoretical ceiling for such a model is the **cell-mean oracle**: assign each cell line
its mean IC₅₀ across all drugs, and use that single number as the prediction for every
(cell, drug) pair. Any model that learns only cell-level features cannot in principle
do better than this oracle, which already has perfect knowledge of every cell's
average sensitivity. If a trained model matches the oracle, it has not learned anything
beyond cell means, regardless of its architecture or input features.

## Experimental design

The dataset is GDSC2: 597 cell lines, 286 drugs, 151,515 (cell, drug) pairs with IC₅₀
measurements. Five omics modalities are available — RNA-seq (19,193 features), somatic
mutations (12,301), CNV (38,590), metabolomics (225), and RPPA (214). All experiments
use a 70/10/20 train/val/test split under three evaluation protocols: **mixed-set**
(random pair assignment), **cell-blind** (test cells unseen during training), and
**drug-blind** (test drugs unseen during training).

The experiment is structured as four stages, each designed to close a specific
alternative explanation for why a model might fail to exceed the ceiling.
Stage 0 computes the cell-mean oracle and sanity-checks the evaluation: if the
per-drug mean predictor scores zero on per-drug r, the metric is not contaminated by
drug-identity leakage.
Stage 1 trains a Ridge regression model on RNA-seq features.
Stage 2A sweeps MLP capacity (Small [128-64-1], Medium [512-256-64-1],
Large [2048-512-128-1]) to test whether Ridge's linearity is the bottleneck.
Stage 2B progressively adds omics modalities (RNA, +mutations, +CNV, +metabolomics,
all five) to test whether RNA-seq alone is the limiting input.
Stage 3 sweeps dropout and weight decay on the cell-blind split, where Ridge
falls short of its oracle, to test whether regularization can close the gap.

## Results

The central finding runs through every stage: on mixed-set and drug-blind evaluation,
every cell-only model converges to per-drug r ≈ **0.644**, matching the cell-mean oracle
(0.644) to within 0.003.
Architecture, capacity, and omics modality are all irrelevant.

### Stage 0: Sanity checks pass

The per-drug mean predictor scores global r = 0.838 on mixed-set, near the theoretical
ceiling for a drug-identity memoriser, while per-drug r is exactly 0 by construction.
Drug-potency signal does not leak into the within-drug ranking metric.

### Stage 1: Ridge reaches the oracle

Ridge on RNA-seq achieves per-drug r = 0.644 ± 0.003 on mixed-set and
0.645 ± 0.008 on drug-blind, both matching the oracle within noise. Hyperparameter
selection is degenerate: all alpha values produce nearly identical validation per-drug r,
and alpha=1000 is chosen in most folds. Regularization becomes irrelevant once the model
stops using drug-specific variance, which is the signature of a model that has collapsed
toward predicting cell means.

### Stage 2A: Capacity does not move the ceiling

Three MLP architectures on RNA-seq (Small, Medium, Large) converge to per-drug r ≈ 0.644
on both mixed-set and drug-blind. Adding layers and parameters provides no benefit over Ridge.

| Model | MS per-drug r | MS global r | CB per-drug r | CB global r | DB per-drug r | DB global r |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| Cell-mean oracle | 0.644 | 0.323 | 0.628 | 0.321 | 0.652 | 0.268 |
| Ridge (RNA-only) | 0.644 ± 0.003 | 0.322 ± 0.003 | 0.463 ± 0.036 | 0.229 ± 0.017 | 0.645 ± 0.008 | 0.344 ± 0.026 |
| MLP Small | 0.642 | 0.323 | 0.420 | 0.214 | 0.651 | 0.268 |
| MLP Medium | 0.644 | 0.323 | 0.425 | 0.217 | 0.650 | 0.267 |
| MLP Large | 0.644 | 0.323 | 0.148 | 0.066 | 0.651 | 0.268 |
| MLP RNA-only | 0.643 | 0.323 | 0.444 | 0.227 | 0.651 | 0.268 |
| MLP RNA+mut | 0.644 | 0.323 | 0.270 | 0.133 | 0.651 | 0.268 |
| MLP RNA+mut+CNV | 0.642 | 0.322 | 0.393 | 0.200 | 0.648 | 0.267 |
| MLP RNA+mut+CNV+met | 0.645 | 0.323 | 0.183 | 0.086 | 0.652 | 0.268 |
| MLP all-5-omics | 0.641 | 0.322 | 0.424 | 0.216 | 0.652 | 0.268 |
| CB reg-sweep best | — | — | 0.429 | 0.219 | — | — |

MS = mixed-set, CB = cell-blind, DB = drug-blind. Ridge: mean ± std over 5-fold CV; MLP: single-seed runs.

### Stage 2B: Additional omics do not help

Progressively adding mutations, CNV, and metabolomics to the Medium MLP produces no
change on mixed-set or drug-blind. Every combination converges to 0.644.
The additional modalities contain no within-drug cell-ranking signal beyond what RNA-seq
already captures.

### Cell-blind: a structurally different problem

Cell-blind evaluation is the exception to the convergence pattern. Ridge achieves only
0.463 ± 0.036, well below the oracle of 0.628 (which
uses test-cell labels and is not a fair target). The capacity sweep reveals that Large MLP
collapses to 0.148 from severe overfitting, while moderate capacity
(Medium at 0.425) fares better. The modality ablation is non-monotonic
and noisy: adding mutations drops performance to 0.270,
suggesting the model memorises modality-specific patterns that do not transfer to unseen
cell lines. The Stage 3 regularization sweep reaches 0.429 at best, below even
the Ridge baseline. Regularization is not the binding constraint; cell-blind performance
is limited by the diversity of cell states available during training.

## Interpretation

The cell-only ceiling is **per-drug r = 0.644**, set by the cell-mean oracle. On the
two evaluation protocols where train and test cells overlap (mixed-set and drug-blind),
no model, whether linear or nonlinear, RNA-only or five-omics, can exceed it. The ceiling
is a structural property of what cell features encode: average cell sensitivity shared
across all drugs. Improving cell representations, adding omics modalities, or increasing
model capacity are not promising directions for improving per-drug r. The experiments in
the next series (`04_cell_representation`) test whether this conclusion holds against more
aggressive alternatives; the experiments in `05_solutions` pursue a different axis entirely.

