# 01 — Drug Feature Null: External Replication

## Question

Does the drug feature null result (Morgan FP Δ ≈ 0 for per-drug r under drug-blind
evaluation) replicate across independent pharmacogenomics datasets?

## Motivation

In GDSC2, Morgan fingerprints add no signal beyond cell-only Ridge under drug-blind
CV (03_drug_feature_null). A sceptic could argue this is specific to the Sanger IC50
assay, GDSC2's drug panel, or its cell line collection. Three independent datasets
break these confounders:

- **CTRPv2**: different institution (Broad), AUC metric, 545 drugs, overlapping cell lines
- **BeatAML**: patient-derived AML samples (not cell lines), AUC, 155 drugs
- **PRISM**: Broad Repurposing Screen, ln_IC50 (same metric as GDSC2), 1415 drugs,
  477 cell lines all present in GDSC2 omics

If Δ ≈ 0 in all three, the finding is platform-independent.

## Experimental design

- **Model**: Ridge(alpha=1.0)
- **Cell features**:
  - CTRPv2/PRISM: RNA PCA(550) + mutation PCA(200) from GDSC2 processed parquets
  - BeatAML: RNA PCA(500) from BeatAML expression (top-5000 variance genes)
- **Drug features**: Morgan FP (radius=2, 2048 bits) vs no drug features
- **CV**:
  - CTRPv2: leave-one-drug-out (LOO) — small drug panel
  - BeatAML: 5-fold drug-blind (KFold, shuffle)
  - PRISM: 10-fold drug-blind (KFold, shuffle) — large panel
- **Metric**: per-drug r (mean Pearson r across drugs in held-out fold)
- **Gate**: |Morgan FP Δ| < 0.01 in all three datasets

## Shared code

- `src/data/ctrpv2.py` — load_ctrpv2_response(), filter_ctrpv2()
- `src/data/beataml.py` — load_beataml_response(), load_beataml_expression()
- `src/data/prism.py` — load_prism()
- `src/utils/ridge.py` — safe_fit_scaler()
- `src/evaluation/per_drug.py` — per_drug_r()

## Reproduce

```bash
uv run python3 experiments/06_external_validation/01_drug_feature_null/jobs/run.py
```

Expected:
- CTRPv2: Morgan FP Δ ≈ 0 (|Δ| < 0.01)
- BeatAML: Morgan FP Δ ≈ +0.003 (matches existing result)
- PRISM: Morgan FP Δ ≈ 0 (|Δ| < 0.01)

Time: ~30 min (all CPU Ridge, no GPU needed).

## Output

`results/run_YYYYMMDD_HHMMSS/results.json`:
```json
{
  "ctrpv2": {"no_drug_r": 0.xxx, "morgan_fp_r": 0.xxx, "delta": 0.xxx, "n_drugs": N},
  "beataml": {"no_drug_r": 0.xxx, "morgan_fp_r": 0.xxx, "delta": 0.xxx, "n_drugs": N},
  "prism":   {"no_drug_r": 0.xxx, "morgan_fp_r": 0.xxx, "delta": 0.xxx, "n_drugs": N}
}
```
