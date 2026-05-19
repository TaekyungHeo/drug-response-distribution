# Representation Alternatives

Can any cell representation break the drug-blind per-drug r ceiling (r=0.645)?

All experiments use Ridge(α=1.0), PASO 10-fold drug-blind CV, no drug features, per-drug Pearson r.

| # | Experiment | Representation | Key result |
|---|-----------|---------------|------------|
| 01 | `01_representation_sweep/` | 6 alternatives: PCA dims, full RNA, pathway | All converge to r≈0.645; Δ≈0.000 |
| 02 | `02_foundation_model/` | scFoundation (50M single-cell pre-trained, 768-dim) | r=0.650 = RNA PCA (Δ=0.000, 534-cell subset) |
| 03 | `03_proteomics_oracle/` | RPPA proteomics (214 proteins) | RNA+RPPA r=0.647 = RNA-only; genuine biological limit |
| 04 | `04_multi_omics/` | RNA+mut + CNV/metabolomics combinations | max Δ=+0.004 (RNA+mut+metabolomics) |

The ceiling is not representation-limited. All encodings converge to r≈0.645 regardless of modality.
