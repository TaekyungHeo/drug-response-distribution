# 02 — Within-MoA Training: External Replication

## Question

Does within-MoA training (leave-one-drug-out within MoA class) improve per-drug r
in datasets other than GDSC2?

## Motivation

In GDSC2, within-MoA LOO yields large per-drug r gains: EGFR signaling +0.375,
ERK MAPK signaling +0.296 (05_solutions/02_training_distribution/01_within_moa).
A sceptic could argue this is GDSC2-specific because GDSC2 overrepresents
kinase inhibitors, or because the IC50 assay is unusually sensitive to MoA
clustering. Testing in CTRPv2 (AUC, Broad), BeatAML (AUC, patient-derived),
and PRISM (ln_IC50, 1415 drugs) addresses this.

MoA annotations come from the Drug Repurposing Hub (CC-BY 4.0, ~6,800 compounds),
which covers 93% of PRISM drugs, 45% of BeatAML drugs, and most CTRPv2 kinase
inhibitors. This replaces manual MoA mapping used in GDSC2 experiments.

## Experimental design

- **Model**: Ridge(alpha=1.0)
- **Cell features**: same as 01_drug_feature_null (RNA PCA + mutations where available)
- **MoA source**: `src/data/repurposing_hub.py` — Drug Repurposing Hub TSV
  (data/processed/repurposing_hub_moa.tsv)
- **CV (all-drug baseline)**: same as 01_drug_feature_null per dataset
- **CV (within-MoA)**: leave-one-drug-out within MoA class
- **Focus MoA classes**: EGFR inhibitor, MEK inhibitor, PI3K inhibitor, mTOR inhibitor
  (select classes with ≥ 3 drugs per dataset after intersection)
- **Gate**: within-MoA per-drug r > all-drug baseline in ≥ 2/3 datasets for
  EGFR inhibitor and MEK inhibitor

## Shared code

- `src/data/ctrpv2.py`, `src/data/beataml.py`, `src/data/prism.py`
- `src/data/repurposing_hub.py` — build_drug_moa_map(), group_by_moa()
- `src/utils/solutions.py` — group_drugs_by_moa()

## Reproduce

```bash
uv run python3 experiments/06_external_validation/02_moa_training/jobs/run.py
```

Expected (GDSC2 reference: EGFR +0.375, ERK +0.296):
- CTRPv2 EGFR: Δ > 0 (existing result: +0.322)
- PRISM EGFR: Δ > 0
- BeatAML: partial coverage — report what's available

Time: ~1 hr (CPU Ridge, LOO loops over MoA classes).

## Output

`results/run_YYYYMMDD_HHMMSS/results.json`:
```json
{
  "datasets": {
    "ctrpv2": {
      "n_drugs_total": N, "n_drugs_with_moa": N,
      "moa_classes": [
        {"moa": "EGFR inhibitor", "n_drugs": N,
         "all_drug_r": 0.xxx, "within_moa_r": 0.xxx, "delta": 0.xxx}
      ]
    },
    "beataml": { ... },
    "prism":   { ... }
  }
}
```
