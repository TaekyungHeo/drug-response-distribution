# PLAN: data preparation

## Purpose

Download and preprocess all raw omics and drug response data into analysis-ready
parquet files. Every downstream experiment depends on these outputs.
Run this once on any new machine before running any experiment.

---

## Data sources

| File | Source | Version / Date | Description |
|------|--------|----------------|-------------|
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | DepMap (figshare) | 24Q4 | RNA-seq TPM log1p |
| `OmicsSomaticMutations.csv` | DepMap (figshare) | 24Q4 | Somatic mutations |
| `OmicsCNGene.csv` | DepMap (figshare) | 24Q4 | Copy number, gene-level log2 |
| `Model.csv` | DepMap (figshare) | 24Q4 | Cell line metadata |
| `CCLE_metabolomics_20190502.csv` | Broad CCLE | 2019-05-02 | 225 metabolites |
| `CCLE_RPPA_20181003.csv` | Broad CCLE | 2018-10-03 | 214 RPPA proteins |
| `GDSC2_fitted_dose_response_24Jul22.csv` | Sanger GDSC2 | Release 8.4 (2022-07-24) | IC50 dose-response |

---

## Outputs

All outputs written to `data/processed/` (git-ignored; regenerate as needed):

| File | Shape | Description |
|------|-------|-------------|
| `rna.parquet` | 1673 × 19193 | RNA-seq, z-scored per gene |
| `mutations.parquet` | 1929 × 12301 | Binary mutation matrix, genes ≥1% prevalence |
| `cnv.parquet` | 1929 × 38590 | CNV log2, NaN→median, z-scored |
| `metabolomics.parquet` | 926 × 225 | Metabolite abundances, z-scored |
| `rppa.parquet` | 894 × 214 | RPPA protein expression, z-scored |
| `drug_response.parquet` | 241578 × 3 | (depmap_id, drug_name, ln_ic50), drugs ≥100 cells |
| `cell_line_index.parquet` | 1972 × 3 | depmap_id ↔ ccle_name ↔ cosmic_id |
| `overlap_cell_lines.parquet` | 597 × 1 | DepMap IDs with all 5 omics + GDSC2 response |

---

## Preprocessing decisions

- **RNA-seq**: columns renamed from `GENE (ENTREZ_ID)` to `GENE`; z-scored across cell lines
- **Mutations**: synonymous variants filtered; pivoted to binary gene×cell matrix;
  genes mutated in <1% of cell lines dropped
- **CNV**: NaN values filled with per-gene median before z-scoring
- **Metabolomics / RPPA**: NaN filled with per-feature median; z-scored
- **Drug response**: Sanger Model ID → DepMap ID via `Model.csv`;
  drugs with <100 tested cell lines excluded (286 drugs pass this threshold)
- **Overlap**: 597 cell lines have all 5 omics modalities **and** appear in GDSC2

---

## How to run

```bash
# Step 1: download raw files (~1.5 GB total, ~30 min on a fast connection)
uv run python3 experiments/00_data_preparation/jobs/download.py

# Step 2: preprocess into parquet (~5 min)
uv run python3 experiments/00_data_preparation/jobs/preprocess.py
```

Idempotent: download skips files that already exist; preprocess overwrites outputs.
Use `--force` to re-download.

---

## Dependencies

- Raw data only (no `data/processed/` inputs)
- Python packages: `pandas`, `numpy`, `requests`, `tqdm`, `pyarrow`
