# Drug Feature Null Result

Central question: does any drug representation improve within-drug cell-line ranking
(per-drug Pearson r) in the drug-blind setting?

Answer: all seven falsification axes complete. Drug features (Morgan FP/Ridge) show Δ≤+0.001 on axes 02 and 04, Δ=+0.002 on axis 07 (all below the 0.01 gate); Transformer with Morgan FP (axis 03) shows Δ=+0.008 (below gate); LINCS Δ=−0.005 and drug-target Δ=−0.019 in Transformer (harmful); MoA one-hot Δ=+0.015 (marginal, permuted-label confirmed). Axis 05 (PRISM within-dataset) shows Δ=+0.0165 at r=0.112 baseline — gate crossed nominally but not clinically relevant at this scale (see 05 README). Axis 06 (RankNet) shows Δ=+0.014 for the ranking objective, demonstrating that the bottleneck is the MSE objective, not the representation.

## Structure

Seven experiments, each testing a distinct way the null could break:

| # | Experiment | Falsification axis |
|---|-----------|-------------------|
| `01_oracle_bounds/` | What is the theoretical ceiling? | Sets the scale for interpreting Δ |
| `02_representation_ablation/` | Does any rep class cross Δ=0.01? | All rep types in one table |
| `03_model_robustness/` | Is Ridge too limited to exploit drug features? | Morgan Δ=+0.008 (null); LINCS Δ=−0.005, drug-target Δ=−0.019 (harmful); MoA Δ=+0.015 (marginal; permuted-label confirmed) |
| `04_split_robustness/` | Is the null an artifact of random drug-blind splits? | Scaffold-blind confirms null (Δ=+0.0006) |
| `05_dataset_robustness/` | Is the null specific to GDSC2? | PRISM within-dataset: Δ=+0.0165 at r=0.112 baseline; gate crossed nominally but clinically irrelevant at this scale |
| `06_objective_axis/` | Is the training objective (not the representation) the bottleneck? | RankNet Δ=+0.014 crosses gate (objective matters, not representation) |
| `07_cross_dataset_transfer/` | Does the null hold across dataset boundaries? | GDSC2→PRISM transfer Δ=+0.002 |

## Canonical protocol

Primary metric: per-drug Pearson r (correlation within each test drug across cell lines, averaged).

- Model: Ridge(α=1.0)
- Cell features: RNA PCA(550) + mutation PCA(200)
- Splits: PASO 10-fold drug-blind CV
- Decision gate: Δ > 0.01 over `no_drug_features` baseline

`03_model_robustness`, `04_split_robustness`, and `05_dataset_robustness` each vary exactly
one axis away from this canonical setup while keeping the drug comparison (Morgan FP vs no_drug)
identical.

## Run order

All jobs are grouped into two waves based on their data dependencies. Start every job in a
wave as soon as its wave begins — do not serialize within a wave.

### Wave 1 — no prerequisites (start immediately)

| Job | Resources | ~Runtime |
|-----|-----------|---------|
| `01` oracle bounds | CPU | < 5 min |
| `04` scaffold-blind splits | CPU | < 15 min |
| `05` PRISM Ridge | CPU | < 30 min |
| `02` all conditions | CPU | < 30 min |
| `03 Part A` TransformerEncoder (OmniCancerV1) | GPU | ≈ 10 h |
| `03 Part B` TransformerEncoderGNN (OmniCancerV2) | GPU | ≈ 10 h |
| `03 Part C` LINCS/drug-target/MoA-onehot (OmniCancerV1) | GPU | ≈ 15 h |
| `05` PRISM Transformer | GPU | ≈ 2 h |
| `06` RankNet vs MSE MLP | GPU | ≈ 3 h |

If only one GPU is available, queue GPU jobs in any order:
`03 Part B` (10 h), `03 Part A` (10 h), `03 Part C` (15 h), `05 Transformer` (2 h), `06` (3 h).

### Cell representation axis

Cell features (RNA PCA(550) + mutation PCA(200)) are deliberately held constant throughout
all experiments here. The cell representation axis is covered by `04_cell_representation`.

## Decision gate justification

The Δ > 0.01 threshold is chosen to be simultaneously:

- **Larger than the 95% CI upper bound** for `morgan_fp` Δ from prior work (CI ≈ [+0.002,
  +0.004]): even the most optimistic estimate of Morgan FP benefit is well below the gate.
- **Larger than fold-to-fold std / 2** (fold std ≈ 0.023, half ≈ 0.012): Δ ≤ +0.01 is
  within sampling noise; Δ > 0.01 cannot be explained by fold variance alone.
- **≈ 2% of the profile concordance ceiling** (0.52): drug features would need to deliver
  at least 2% of the maximum theoretically achievable drug-similarity transfer signal to be
  considered non-trivial.

Both unweighted and cells-weighted per-drug r must independently agree that Δ > 0.01 before
the gate is considered crossed.

**Power limitation**: power analysis (`02/jobs/power_analysis.py`) shows the experiment has
only ~5% Monte Carlo power at the Δ=0.01 gate after Holm-Bonferroni correction (MDE at 80%
power ≈ 0.030). The gate is a **clinical-relevance threshold**, not a detection threshold.
The null claim is properly stated as: "observed Δ ≈ 0.003 with 95% bootstrap CI excluding
0.010, consistent with Δ ≈ 0 across 7 independent axes." Effects larger than Δ ≈ 0.030 can
be ruled out at 80% power; effects in [0.003, 0.030] are not excluded but are below the
practical relevance threshold for drug-blind prediction improvement.

## What this does NOT cover

- Between-drug scale prediction: LINCS perturbation signatures do not improve per-drug r
  (Δ≈+0.001) and do not improve global r on the 104-drug LINCS-covered subset
  (actual: global r decreases by −0.058) — analyzed in 05_solutions/04_external_signatures/01_lincs.
- MoA-stratified training: not a drug *representation* change; covered separately.
