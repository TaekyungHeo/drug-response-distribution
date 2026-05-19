# 03 — K=1 Mechanism: Patient-Specific Signal at a Single Observation

## Research question

Does K=1 in patient-derived BeatAML data carry patient-specific profile information,
unlike GDSC2 where K=1 is near-pure potency calibration?

## Background

In GDSC2, the K=1 dip below K=0 arises because a single cell-line measurement
is dominated by potency calibration rather than cell-ranking signal: it finds
training drugs with similar mean potency but not similar cell-response profiles
(03_few_shot/01_response_matching). BeatAML patients have higher biological
heterogeneity than GDSC2 cell lines. If K=1 in patient data selects training drugs
with profiles similar at that specific patient, it carries patient-specific information —
a stronger signal than potency calibration alone.

This is an extension finding (not a GDSC2 replication): it shows that K=1 informativeness
is context-dependent, and that the K-shot mechanism operates differently in patient-derived
vs cell-line data.

## Experimental design

- **Data**: BeatAML (dbGaP phs001657), 155 drugs, 520 patients
- **CV**: 5-fold drug-blind, 30 random patient choices per drug
- **Method A** (cell-specific): nearest training drug by response distance at the observed patient
- **Method B** (mean potency): nearest training drug by distance to mean response across all patients
- **Gate**: Δ(A − B) > 0 (K=1 carries patient-specific information beyond potency)

## Results

| Method | Per-drug r |
|--------|:----------:|
| A: Cell-specific matching | 0.3560 |
| B: Mean potency matching | 0.2980 |
| **Δ(A − B)** | **+0.0590** |

30 random patient draws per drug, averaged.

## Interpretation

Method A exceeds Method B by +0.0590, confirming
that K=1 in BeatAML carries patient-specific profile information beyond potency calibration.
This contrasts with GDSC2 cell lines where K=1 is dominated by potency (and produces a dip
below K=0). The difference reflects the higher biological heterogeneity of patient samples:
a patient's response to one drug is more predictive of their response to mechanistically
similar drugs than the same measurement is in a cell line.


