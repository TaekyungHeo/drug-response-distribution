# 03 — K-Shot Response Matching: External Replication

## Question

Does the K-shot response matching improvement (K-curve lift, K=1 dip) replicate
across independent datasets?

## Motivation

In GDSC2, K-shot response matching raises per-drug r from 0.637 (K=0) to 0.713
(K=50), and K=1 produces a characteristic dip below K=0 because a single
observation is dominated by potency calibration rather than cell-ranking signal
(05_solutions/03_few_shot/01_response_matching). Testing in three external datasets
breaks dataset-specific confounders.

**Important setup distinction**:
- CTRPv2 / BeatAML: train on GDSC2 drugs, predict on held-out drugs in
  target dataset. K pilot observations come from the target dataset's cells/patients.
  Tests whether matching transfers to new cell populations.
- PRISM: K pilot observations on PRISM cell lines (all in GDSC2 omics). Train on
  non-held-out PRISM drugs. Tests whether matching works across a much larger drug
  panel on the same cell infrastructure.

Both test the matching mechanism; they validate different transfer axes.

## Experimental design

- **Model**: Ridge(alpha=1.0) for base predictions; response matching for refinement
- **K values**: [0, 1, 3, 5, 10, 20, 50]
- **Matching**: top-5 nearest training drugs by response profile (Pearson over K obs)
  Blend weight w = 0.5 (fixed, as in 05_solutions/03_few_shot)
- **CV**:
  - CTRPv2: 5-fold drug-blind (small panel)
  - BeatAML: 5-fold drug-blind, 20 random patient draws per drug
  - PRISM: 10-fold drug-blind
- **Metric**: per-drug r

**Gates**:
1. K=50 > K=0 in all three datasets (lift replicates)
2. K=1 < K=0 in BeatAML (dip replicates — expected from single-patient potency noise)
3. K=1 < K=0 in CTRPv2 (dip replicates in cell line context)

## PRISM-specific note

PRISM K-shot tests a complementary axis: robustness to much wider drug diversity
(1415 drugs) while cell features are fully available (no OOD cells). A positive
result here is evidence the matching mechanism is not brittle to drug panel scale.

## Shared code

- `src/data/ctrpv2.py`, `src/data/beataml.py`, `src/data/prism.py`
- `src/utils/response_matching.py` — existing response matching utilities
- `src/evaluation/per_drug.py` — per_drug_r()

## Reproduce

```bash
uv run python3 experiments/06_external_validation/03_kshot_matching/jobs/run.py
```

Expected:
- CTRPv2 K=0: ~0.41, K=50: > K=0
- BeatAML K=0: ~0.46, K=1 < K=0, K=50: ~0.60
- PRISM K=0: TBD, K=50: > K=0

Time: ~2 hr (BeatAML random draws are slow).

## Output

`results/run_YYYYMMDD_HHMMSS/results.json`:
```json
{
  "ctrpv2": {
    "k_curve": [{"k": 0, "per_drug_r": 0.xxx}, ...],
    "n_drugs": N, "n_cells": N
  },
  "beataml": {
    "k_curve": [{"k": 0, "per_drug_r": 0.xxx}, ...],
    "n_drugs": N, "n_patients": N
  },
  "prism": {
    "k_curve": [{"k": 0, "per_drug_r": 0.xxx}, ...],
    "n_drugs": N, "n_cells": N,
    "note": "same cells as GDSC2 training — tests drug panel breadth"
  }
}
```
