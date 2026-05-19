# Cell Representation

Central question: is the drug-blind per-drug r ceiling (r≈0.645) a genuine biological
limit, or an artifact of the cell representation, model class, or methodology?

## Paper mapping

This experiment group does not have its own results section in the paper. Its results
appear in two places:

- **Fig. 2c and §2 (drug feature null)**: "An omics ablation confirmed that RNA-seq
  alone achieves per-drug r = 0.645, and that mutations and further modalities add
  ≤+0.004 across six cell feature sets. The bottleneck is not cell-representation
  expressiveness." This covers the multi-omics ablation from
  02_representation_alternatives and 04_methodological_robustness.
- **§5 (external validation)**: "Cell-blind CV falls substantially below drug-blind
  (r = 0.438 vs 0.645), inverting the conventional ranking." This comes from
  01_ceiling_characterization/01_split_ceilings.
- **Supplementary Note 3** (baselines and performance ceilings): replicate concordance
  ceiling (0.754), cell-blind vs drug-blind, and the cell-mean prior from
  01_ceiling_characterization/04_measurement_noise and 01_split_ceilings.
- **Supplementary Note 12** (extended methods): extended description of cell
  representation alternatives.

## Structure

| Group | Question | Status |
|-------|----------|--------|
| `01_ceiling_characterization/` | What is the ceiling and is it real? | Complete (4/4) |
| `02_representation_alternatives/` | Can better representations break the ceiling? | Complete (4/4) |
| `03_data_sufficiency/` | Is the ceiling data-limited or information-theoretic? | Complete (1/1) |
| `04_methodological_robustness/` | Is the ceiling Ridge/MSE/random-split-specific? | Complete (3/3) |

## Canonical setup

Per-drug Pearson r, 10-fold drug-blind CV (PASO splits), Ridge(α=1.0),
no drug features. Baseline: RNA PCA(550) + mutation PCA(200), r=0.645.

## Logical argument assembled

```
r=0.645 is real (not a metric artifact):
  01 → 01_split_ceilings: drug-blind r=0.645 > cell-blind r=0.438; drug-mean oracle
       global_r=0.845 but per-drug r≈0; cell-mean prior matches Ridge per-drug r
  01 → 02_lineage_analysis: all 7 lineages r≥0.48; LINCS-covered (0.665) vs uncovered (0.628), Δ=0.037
  01 → 03_within_lineage_training: within-lineage r ≈ pan-cancer per-lineage r (Δ≤0.03)
  01 → 04_measurement_noise: replicate r=0.754; r=0.645 is 86% of noise ceiling

r=0.645 is not representation-limited:
  02 → 01–04: RNA PCA, scFoundation, RPPA, CNV+metabolomics all Δ≤+0.004

r=0.645 is approaching information-theoretic limit:
  03 → 01_learning_curve: r rises from 0.41 (10% data) to 0.645 (100%); rate decreasing
       but no clear plateau before full dataset — ceiling may have data-limited component

r=0.645 is not method-limited:
  04 → 01_nonlinear_models: XGBoost r=0.645 (Δ≈0), MLP r=0.639 (Δ=-0.006)
  04 → 02_chemical_split: Tanimoto split r=0.660 (Δ=+0.015 leakage, negligible)
  04 → 03_ranking_loss: ranking loss r=0.648 (Δ=+0.003 vs MSE, not significant)

Conclusion: r=0.645 is the practical ceiling of drug-blind prediction from transcriptomics
under standard cell-line training conditions. The 14% gap to the measurement ceiling
(0.754) reflects a combination of biological noise and data coverage limits.
```

## Run order

`01_ceiling_characterization/01_split_ceilings` →
remaining experiments in any order.
