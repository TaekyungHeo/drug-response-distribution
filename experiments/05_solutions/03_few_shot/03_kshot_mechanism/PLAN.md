# 04_beataml_k1_mechanism

## Question
Does K=1 in patient data carry patient-specific profile information, unlike GDSC2 where K=1 is pure potency calibration?

## Method
BeatAML (155 drugs, 520 patients). 5-fold drug-blind CV, 30 random patient choices per drug. Compare: (A) cell-specific matching — nearest training drug by response distance at the observed patient; (B) mean potency matching — nearest training drug by distance to mean response. If A > B, K=1 carries patient-specific information.

## Key Result
Method A r=0.356, Method B r=0.298, Δ(A−B)=+0.059. K=1 patient observation carries patient-specific profile information — unlike GDSC2 cell lines where K=1 is near-pure potency calibration.

## Reproduce
```bash
~/.local/bin/uv run python3 experiments/07_external_validation/04_beataml_k1_mechanism/jobs/run.py
```
Expected: cell-specific ≈ 0.356, mean potency ≈ 0.298, Δ ≈ +0.059. Time: ~20 min.

## Dependencies
- Data: BeatAML (dbGaP phs001657)
- Prior: `03_beataml_validation/` (establishes K=1 lift)

## Context in Paper
Results: "K=1 in patient data provides more than potency calibration — patient heterogeneity is informative at a single observation".
