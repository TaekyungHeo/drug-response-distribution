# 03 — RPPA Proteomics as Cell Representation

Tests whether CCLE RPPA proteomics (214 proteins) can improve the per-drug r ceiling
established by RNA-seq. Addresses whether the RNA-based ceiling is a genuine biological
limit or an artifact of representation type.

**Key result:** RPPA alone (r=0.557) performs worse than RNA (r=0.647). RNA+RPPA
concatenated (r=0.647) matches RNA alone exactly. Adding protein-level measurements
from the same cells provides no additional predictive signal beyond transcriptomics.

## Conditions (restricted to 588 cells with RPPA coverage)

| Condition | Features | Per-drug r |
|-----------|----------|-----------|
| A | RNA PCA(550) + Mut PCA(200) | 0.6469 |
| B | RPPA (214 proteins, raw) | 0.5568 |
| C | RNA + Mut + RPPA | 0.6469 |

Note: overall r values are restricted to the 588-cell RPPA-covered subset and differ
slightly from the full 687-cell pan-cancer baseline (0.645).

## Input data

- `data/raw/CCLE_RPPA_20181003.csv` — RPPA protein levels (index: CELLLINE_TISSUE format)
- `data/processed/rna.parquet`, `mutations.parquet`, `cell_line_index.parquet`
- `external/PASO/data/10_fold_data/drug_blind/DrugBlind_{train,test}_Fold{0..4}.csv`

## Output files

- `report/data/results_{A,B,C}_*.json` — per-condition per-drug r summaries
