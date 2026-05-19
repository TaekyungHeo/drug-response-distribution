# PLAN: Proteomics Oracle — Does RPPA Break the Per-Drug r Ceiling?

## Hypothesis

If the per-drug r ceiling reflects a genuine biological limit of predictability from
transcriptomics, then adding protein-level measurements (RPPA) should not materially
improve predictions. The claim "Apoptosis is a genuine biological limit" is tested with
RPPA: if RPPA lifts the Apoptosis per-drug r by >0.05, the claim must be revised to
"RNA-representation limit" (not biological).

Prediction: RNA+RPPA ≈ RNA alone (Δ≈0 overall and for Apoptosis), confirming the
biological limit claim.

## Design

- **Model**: Ridge(α=1.0), no drug features
- **Cell pool**: RNA ∩ mutations ∩ RPPA coverage (~588 cells)
- **Splits**: PASO 10-fold drug-blind CV (restricted to RPPA-covered cells)
- **Primary metric**: mean per-drug Pearson r (overall + per-MoA)
- **RPPA**: CCLE RPPA 214 proteins, no PCA (fewer features than RNA)

| Condition | Cell features | Expected Δ vs A |
|-----------|--------------|----------------|
| `A_rna_mut` | RNA PCA(550) + mut PCA(200) | 0 (reference) |
| `B_rppa` | RPPA 214 proteins | −0.08 (RPPA worse than RNA) |
| `C_rna_mut_rppa` | RNA + mut + RPPA concat | ≈0 |

**Per-MoA analysis**: Apoptosis regulation, ERK MAPK signaling, EGFR signaling,
PI3K/MTOR signaling, Mitosis.

**Verdict logic**:
- RPPA lifts Apoptosis Δ > 0.05 → revise to "RNA-representation limit"
- |Δ| ≤ 0.02 → "genuine biological limit" claim strengthened
- 0.02 < Δ ≤ 0.05 → marginal, interpret with caution

## How to run

```bash
# Local
uv run python3 experiments/04_cell_representation/02_representation_alternatives/05_proteomics_oracle/jobs/run.py

# DGX cluster (from spark1)
sbatch experiments/04_cell_representation/02_representation_alternatives/05_proteomics_oracle/jobs/sbatch.sh
```

Expected runtime: ~15 min (10 folds, 3 conditions, MoA stratification)

## Validation checks

- `A_rna_mut` per-drug r ≈ 0.632 (restricted to RPPA cells, slightly lower than full set)
- `B_rppa` per-drug r ≈ 0.547 (RPPA alone worse than RNA)
- `C_rna_mut_rppa` per-drug r ≈ 0.632 (no improvement over RNA)
- Apoptosis Δ (RPPA vs RNA) ≤ 0.02

## Output

`report/data/results.json` — schema:
```json
{
  "summary": {
    "A_rna_mut":      {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_A": 0.0},
    "B_rppa":         {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_A": float},
    "C_rna_mut_rppa": {"per_drug_r_mean": float, "per_drug_r_std": float, "delta_vs_A": float}
  },
  "moa_summary": {
    "A_rna_mut": {"Apoptosis regulation": {"mean": float, "delta_vs_A": float}, ...},
    ...
  },
  "apoptosis_verdict": "string",
  "fold_results": { ... }
}
```

## Dependencies

- `data/processed/rna.parquet`, `data/processed/mutations.parquet`
- `data/processed/cell_line_index.parquet`
- `data/raw/CCLE_RPPA_20181003.csv` (raw RPPA, CELLLINE_TISSUE index)
- `external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv` (MoA labels)
- PASO drug-blind splits: `external/PASO/data/10_fold_data/drug_blind/`
- `src/evaluation/per_drug.per_drug_r`
- `src/utils/paso_folds.load_paso_pairs`
- `src/utils/ridge.{compress_cell, safe_fit_scaler}`

## Resources

- `--cpus-per-task=8`
- `--mem=32G` (RPPA CSV ~5MB; peak <500MB)
- `--time=2:00:00`
- No GPU needed
