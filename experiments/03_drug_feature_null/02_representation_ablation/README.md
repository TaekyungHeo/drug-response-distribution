# 02 — Representation Ablation

## Research question

Does any drug representation class — structural, functional, mechanistic, or pharmacological
— improve within-drug cell-line ranking (per-drug r) in the drug-blind setting?

## Design

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
| morgan_fp | 0.646 ± 0.025 | +0.0009 [+0.0004, +0.0015] | 0.0049 |
| chemberta (PCA-64) | 0.646 ± 0.024 | +0.0009 | 0.0094 |
| chembl_targets | 0.646 ± 0.024 | +0.0007 | 0.0094 |
| lincs (PCA-64, 104 drugs) | 0.665 ± 0.038 | +0.0005 | 0.0094 |
| all_concat | 0.646 ± 0.024 | +0.0009 | 0.0049 |
| morgan_fp_shuffled (degenerate) | 0.646 ± 0.024 | +0.0009 | — |
| random_continuous (degenerate) | 0.646 ± 0.024 | +0.0009 | — |

**Decision gate: delta > 0.01. No representation crosses the gate.**

Note: GNN embeddings (from 03_model_robustness Part B) and PRISM drug features were evaluated
during analysis but not incorporated into the final table: GNN fits under "learned" in the
interpretation below and PRISM drug features are covered under axis 05 (dataset robustness).
The paper's Supplementary Note 5 (drug feature null: full representation sweep) reports the
five conditions above plus MoA one-hot (from 05_solutions/02_training_distribution/03_onehot_control).
Supplementary Note 4 (per-drug performance and delta by pathway) reports the per-drug delta
distribution and pathway-stratified delta table.

### Degenerate baseline check

Shuffled Morgan FP and random continuous vectors produce the same delta (+0.0009) as
real Morgan FP (+0.0009). This is a critical finding: adding ANY extra features to Ridge
— even random noise — produces the same tiny positive delta due to the mechanical effect
of additional input dimensions on Ridge's bias-variance tradeoff. The "real" drug feature
signal is indistinguishable from noise.

### Alpha sensitivity

| alpha | morgan_fp per-drug r |
|:-----:|:---:|
| 0.01 | 0.6462 |
| 0.1 | 0.6462 |
| 1.0 | 0.6462 |
| 10.0 | 0.6462 |
| 100.0 | 0.6461 |

Per-drug r is invariant to alpha across 4 orders of magnitude, ruling out the hypothesis
that alpha=1.0 suppresses drug feature coefficients.

### Drug feature scaling

| Scale | delta |
|:-----:|:---:|
| 0.1x | +0.0007 |
| 0.3x | +0.0009 |
| 1.0x | +0.0009 |
| 3.0x | +0.0009 |
| 10.0x | +0.0009 |

Scaling drug features by 10x has no effect. The model is completely insensitive to
drug feature magnitude.

### Similarity-stratified delta

| Tanimoto similarity to nearest training drug | n_drugs | delta |
|:---:|:---:|:---:|
| Low (0.0-0.3) | 154 | +0.0008 |
| Mid (0.3-0.5) | 40 | +0.0011 |
| High (0.5-1.0) | 39 | +0.0013 |

Delta is marginally larger for drugs with higher structural similarity to training drugs,
but the range (0.0008-0.0013) is negligible. Even for the most structurally novel drugs,
Morgan FP provides no meaningful benefit.

### Per-drug delta distribution

233 drugs tested. Summary statistics:
- Mean delta: +0.0009
- Range: [-0.014, +0.018]
- 97% of drugs show |delta| <= 0.01
- Largest positive: GW441756 (+0.018), Savolitinib (+0.016), Methotrexate (+0.016)
- Largest negative: OSI-027 (-0.014), UMI-77 (-0.009)

The distribution is unimodal and centered near zero. No subgroup of drugs shows
consistent benefit from drug features.

### LINCS note

LINCS shows per-drug r = 0.665 in this experiment (n=104 drugs across all 10 folds).
The higher absolute value vs the 0.645 whole-dataset baseline reflects drug selection
bias (LINCS-covered drugs tend to be well-characterized with higher baseline per-drug r).
The delta (+0.0005, computed against the matched no-drug baseline of 0.665) is smaller
than Morgan FP's.

Note: The paper's canonical LINCS values come from the dedicated experiment in
`05_solutions/04_external_signatures/01_lincs`, which uses a proper matched comparison
(no_drug matched = 0.659, LINCS = 0.660, Δ = +0.001; global r Δ = −0.058). The
qualitative conclusion is identical: LINCS does not improve per-drug r.

## Interpretation

1. **No drug representation crosses the decision gate (delta > 0.01).** Structural
   (Morgan FP), learned (ChemBERTa), mechanistic (ChEMBL targets), and pharmacological
   (LINCS) representations all fail. GNN-based models are tested in axis 03
   (03_model_robustness; Δ=+0.0076, below gate).

2. **The delta is indistinguishable from noise.** Real Morgan FP delta (+0.0009) equals
   shuffled Morgan FP delta (+0.0009) and random vector delta (+0.0009). Drug features
   are not being used for drug-specific prediction.

3. **Power caveat**: The experiment has ~5% power at the gate (delta=0.01) and MDE ~0.030
   at 80% power. We cannot rule out effects smaller than 0.03 with high confidence. However,
   the degenerate baseline equivalence provides stronger evidence than the gate alone: the
   model demonstrably ignores drug feature content.

4. **Relative to ceiling**: Tanimoto concordance per-drug r = 0.718 (from 01_oracle_bounds).
   Drug features extract 0.0009 / 0.718 = 0.13% of the available similarity-transfer signal.

## Validation checks

- no_drug per-drug r = 0.645 ± 0.025: consistent with expected ~0.631 ± 0.023 (difference
  likely due to 10-fold vs 5-fold CV and cell population)
- Degenerate baselines recover no_drug r: **PASS**
- All conditions delta < 0.01: **PASS**
- morgan_fp delta CI [+0.0004, +0.0015] excludes zero but not gate: **PASS**
- Alpha sensitivity flat: **PASS**
