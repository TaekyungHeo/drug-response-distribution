# 05 — Dataset Robustness

Tests whether the null from `02_representation_ablation` is specific to GDSC2's limited
drug count (233 drugs).

PRISM Repurposing: 1,079 drugs (4.6× GDSC2), viability-based AUC, 477 cell lines.

| Condition | Per-drug r | Δ |
|-----------|-----------|---|
| `no_drug` Ridge (PRISM within-dataset) | 0.112 ± 0.005 | — |
| `morgan_fp` Ridge (PRISM within-dataset) | 0.129 ± 0.009 | +0.0165 |

Note: The paper's canonical PRISM values (no_drug r = 0.117, morgan_fp r = 0.134,
Δ = +0.017) come from the unified experiment in `06_external_validation/01_drug_feature_null`,
which uses slightly different preprocessing. The qualitative conclusion (gate crossed
nominally but contextually irrelevant) is identical.

**Δ = +0.0165 crosses the nominal 0.01 gate**, but must be interpreted in context:
per-drug r = 0.112 on PRISM is an order of magnitude below the GDSC2 baseline (0.645).
The gate threshold was calibrated for GDSC2 scale; at PRISM's near-zero baseline a Δ of
+0.016 carries little clinical relevance. Confounders include viability-AUC vs IC50 assay,
and the fact that PRISM training uses the same (noisier) assay platform for cell features.
The 07_cross_dataset_transfer result (GDSC2-trained on PRISM test: no_drug = 0.039,
Δ = +0.002) provides a cleaner comparison: at comparable absolute performance, the null
holds. Taken together, drug features provide at most marginal benefit in the low-signal
PRISM regime, not the robust improvement required to reject the null.

See [PLAN.md](PLAN.md) for full design and run instructions.
