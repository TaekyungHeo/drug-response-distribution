# 06 — External Validation

Do the GDSC2 findings replicate in independent pharmacogenomics datasets?

## Core question

GDSC2 yields four findings:
1. **Drug feature null**: Morgan FP Δ ≈ 0 for per-drug r under drug-blind evaluation
2. **Within-MoA training**: LOO within MoA class raises per-drug r substantially
3. **K-shot response matching**: K-curve lift from K=0 to K=50; gains meaningful above K≥20 (crossover threshold). Scope condition: requires functional drug analogs in the training set.
4. **LINCS decomposition**: LINCS does not improve per-drug r (Δ≈+0.001) and reduces global r on the LINCS-covered subset

External datasets used:

| Dataset | Type | Metric | Drugs | Cells/patients |
|---------|------|--------|-------|----------------|
| CTRPv2 | Cell lines (Broad) | AUC | 545 | 812 |
| BeatAML | Patient-derived AML | AUC | 155 | 520 |
| PRISM | Cell lines (Broad, Repurposing) | AUC | 1079 | 477 |

PRISM uses the same cell lines as GDSC2 (all 477 present in GDSC2 omics),
making it the cleanest control for drug panel effects (cell population held constant;
only drug panel varies). CTRPv2 and BeatAML test OOD cell populations and AUC response.

## Experiments

| # | Experiment | Finding validated | Datasets |
|---|-----------|-------------------|---------|
| 01 | `01_drug_feature_null` | Drug feature null (Δ≈0 per-drug r): CTRPv2 Δ=+0.005, BeatAML Δ=+0.001, PRISM Δ=+0.017 (marginal; low baseline) | CTRPv2, BeatAML, PRISM |
| 02 | `02_moa_training` | Within-MoA training: CTRPv2 EGFR Δ=+0.371 (8 drugs), RAF Δ=−0.004 (null); class-specific pattern replicates | CTRPv2, BeatAML, PRISM |
| 03 | `03_kshot_matching` | K-shot K-curve: CTRPv2 0.411→0.463 (+0.051 at K=50); BeatAML 0.453→0.521 (+0.067 at K=50); **PRISM: scope failure** — curve collapses at K≥3 (no functional drug analogs in GDSC2 training set) | CTRPv2, BeatAML, PRISM |

**Finding 4 (LINCS)**: CTRPv2 LINCS coverage is 14/545 drugs (2.6%); BeatAML has no
LINCS signatures. External validation of the LINCS finding is not feasible with
available data and is noted as a limitation.

**K=1 mechanism extension** (patient-specific profile at K=1): moved to
`05_solutions/03_few_shot/03_kshot_mechanism` — it is a new finding in BeatAML,
not a GDSC2 replication.

## MoA annotations

Experiment 02 uses the **Drug Repurposing Hub** (Broad Institute, CC-BY 4.0,
~6,800 compounds) stored at `data/processed/repurposing_hub_moa.tsv`.
Coverage: PRISM 93%, BeatAML 45%, CTRPv2 kinase inhibitors ~100%.

## Shared code

- `src/data/ctrpv2.py` — CTRPv2 loading and cell matching
- `src/data/beataml.py` — BeatAML response and expression loading
- `src/data/prism.py` — PRISM loading (existing)
- `src/data/repurposing_hub.py` — Drug Repurposing Hub MoA annotations

## Running all experiments

All three experiments are CPU-only Ridge (no GPU required):

```bash
uv run python3 experiments/06_external_validation/01_drug_feature_null/jobs/run.py  # ~30 min
uv run python3 experiments/06_external_validation/02_moa_training/jobs/run.py       # ~1 hr
uv run python3 experiments/06_external_validation/03_kshot_matching/jobs/run.py     # ~2 hr
```

