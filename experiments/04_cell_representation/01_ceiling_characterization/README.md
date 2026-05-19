# Ceiling Characterization

What is the cell-side prediction ceiling, and is it a genuine limit?

| # | Experiment | Question | Key result |
|---|-----------|---------|------------|
| 01 | `01_split_ceilings/` | Drug-blind vs cell-blind per-drug r; cell-mean prior diagnostic | Drug-blind r=0.645; cell-blind r=0.438; cell-mean prior matches Ridge (confirms ridge learns toxicity-score) |
| 02 | `02_lineage_analysis/` | Is pan-cancer r an artifact of lineage composition? | All 7 lineages r≥0.48; LINCS-covered (0.665) vs uncovered (0.628), Δ=0.037 |
| 03 | `03_within_lineage_training/` | Does within-lineage training match pan-cancer per-lineage r? | Yes — within-lineage r matches pan-cancer per-lineage r (Δ≤0.03 per lineage); no cross-lineage inflation |
| 04 | `04_measurement_noise/` | What is the within-assay replicate ceiling? | mean r_yy=0.754 ± 0.124 (9 anchor drugs, 6,288 replicate pairs) |

Key finding (from 01-04): drug-blind r=0.645 is 86% of the measurement ceiling (0.754); lineage composition and cross-lineage inflation are ruled out.
