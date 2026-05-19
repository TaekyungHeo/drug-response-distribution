# 03 — Model Robustness

Tests whether the null from `02_representation_ablation` is a Ridge capacity artifact.

**Part A**: TransformerEncoder (4L×256d; referred to as OmniCancerV1 internally), Morgan FP vs no_drug, 10-fold drug-blind.

| Condition | Per-drug r | Δ |
|-----------|-----------|---|
| `no_drug` (Transformer) | 0.6448 | — |
| `morgan_fp` (Transformer) | 0.6524 | +0.008 |

**Null holds in a nonlinear model with 21M parameters.** Δ=+0.008 is below the 0.01
gate. Fold-level deltas range from −0.018 to +0.032 (high variance); no degenerate
control (shuffled fingerprints) was run, so the +0.008 delta may be a dimensionality
artifact rather than drug-content signal.

**Part B** (secondary): trains TransformerEncoderGNN (GCN drug encoder; referred to as OmniCancerV2 internally) on drug-blind splits to
extract 256-dim GNN embeddings for use as the `gnn` condition in `02_representation_ablation`.

**Part C**: Extended representation ablation in the same OmniCancerV1 Transformer. Tests
LINCS, drug-target, and MoA one-hot representations under 10-fold PASO drug-blind CV.

| Condition | Type | Dim | Per-drug r Δ | Notes |
|-----------|------|-----|:---:|-------|
| `lincs_pca64` | Functional, continuous | 64 | −0.005 (4/10 folds positive, Wilcoxon P=0.625) | LINCS L1000 consensus, 104 matched drugs only |
| `drug_target` | Mechanistic, sparse binary | ~5145 → PCA(256) | −0.019 (0/10 folds positive, Wilcoxon P=0.002) | ChEMBL drug-target matrix; actively harmful |
| `moa_onehot` | Mechanistic, categorical | 24 | +0.015 (7/10 folds positive, Wilcoxon P=0.053) | PASO Target Pathway one-hot |

Permuted-MoA control: Δ=−0.012 vs no-drug baseline (P=0.002, 0/10 folds positive);
real vs permuted MoA difference significant (P=0.014, 9/10 folds positive). Confirms
the Transformer exploits MoA class identity, not dimensionality reduction.

**Null holds for continuous/binary representations; MoA categorical gives marginal gain.**
The drug-target result (Δ=−0.019) demonstrates that high-dimensional sparse features
actively distract the model. MoA Δ=+0.015 is an order of magnitude smaller than
within-MoA training gains for targeted classes (ERK MAPK +0.296, EGFR +0.375).

See [PLAN.md](PLAN.md) for full design and run instructions.
