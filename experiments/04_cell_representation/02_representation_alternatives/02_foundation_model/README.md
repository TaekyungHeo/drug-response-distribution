# 02 — scFoundation (768-dim) Cell Embeddings as Ridge Features

Tests whether scFoundation — a foundation model pretrained on 50M single cells producing
768-dim embeddings — breaks the per-drug r ceiling established by RNA-PCA(550)+Mut-PCA(200).

**Key result:** scFoundation per-drug r = 0.650, identical to RNA-PCA baseline = 0.650
(Δ = 0.000). Verdict: foundation model converges to the same
function as bulk RNA-PCA — the ceiling is not representation-limited.

## Conditions (restricted to 534 cells with scFoundation coverage)

| Condition | Features | Per-drug r |
|-----------|----------|------------|
| A | RNA PCA(550) + Mut PCA(200) | 0.650 |
| B | scFoundation (768-dim) | 0.650 |
| C | RNA PCA + scFoundation concat | 0.650 |

## Input data

- `data/external/scFoundation/50M-0.1B-res_embedding.npy` — (561, 768) embeddings
- `data/external/scFoundation/cancer_cell_line.info` — cell line names
- `data/processed/rna.parquet`, `mutations.parquet`, `cell_line_index.parquet`
- `external/PASO/data/10_fold_data/drug_blind/DrugBlind_{train,test}_Fold{0..4}.csv`

## Obtaining scFoundation embeddings

The pre-computed cell-line embeddings are **not tracked in git** (large binary, not redistributable without permission). Two options:

**Option A — Download pre-computed embeddings from the scFoundation authors**

```bash
# 1. Request access or download from the scFoundation repository:
#    https://github.com/biomap-research/scFoundation
#    The file is listed under "Resources" → "Cancer cell line embeddings"
#
# 2. Place files at:
mkdir -p data/external/scFoundation
# Copy 50M-0.1B-res_embedding.npy  → data/external/scFoundation/
# Copy cancer_cell_line.info        → data/external/scFoundation/
```

**Option B — Recompute embeddings yourself**

```bash
# Clone scFoundation and run inference on CCLE RNA-seq data:
#   https://github.com/biomap-research/scFoundation
# Input: GDSC2/CCLE bulk RNA-seq (TPM log1p), gene list must match scFoundation vocab
# Output: (N_cells, 768) embedding matrix saved as .npy
# Expected shape: (561, 768) for the 561 cell lines that overlap with our GDSC2 set
```

The `cancer_cell_line.info` file maps DepMap IDs (ACH-XXXXXX) to cell line names and cancer types (tab-separated, one cell per row). This can be reconstructed from DepMap's `sample_info.csv`.

## Output files

- `results/run_<timestamp>/results.json` — per-condition summary, fold-level prediction
  correlation between A and B, verdict string
