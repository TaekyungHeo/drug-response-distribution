# Data Preparation

Every experiment in this repository reads from `data/processed/`. This directory
is git-ignored and must be built once on each machine before running anything else.

Two steps: **download** raw files, then **preprocess** them into parquet.

---

## Quick start

```bash
# From the repository root:
uv run python3 experiments/00_data_preparation/jobs/download.py
uv run python3 experiments/00_data_preparation/jobs/preprocess.py
```

Both steps are idempotent. Download skips files already present; preprocess
overwrites `data/processed/` outputs. Re-download with `--force` if needed.

---

## What data

This project uses three public databases:

### DepMap 24Q4 — cell line omics
The Cancer Dependency Map (Broad Institute) releases quarterly snapshots of
cancer cell line molecular profiles. We use the 24Q4 release (figshare, DOI
10.6084/m9.figshare.22765112).

| File | What it contains |
|------|-----------------|
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | RNA-seq gene expression: 1,673 cell lines × 19,193 genes, TPM log1p |
| `OmicsSomaticMutations.csv` | Somatic mutation calls for 1,929 cell lines |
| `OmicsCNGene.csv` | Copy number (log2 relative) for 1,929 cell lines × 25,368 genes |
| `Model.csv` | Cell line metadata: DepMap ID, CCLE name, COSMIC ID, tissue, etc. |

### CCLE legacy — proteomics
Two older CCLE datasets not yet superseded in DepMap quarterly releases:

| File | What it contains |
|------|-----------------|
| `CCLE_metabolomics_20190502.csv` | 225 metabolite abundances (log10) for 926 cell lines |
| `CCLE_RPPA_20181003.csv` | 214 RPPA antibody measurements for 894 cell lines |

### GDSC2 release 8.4 — drug response
The Genomics of Drug Sensitivity in Cancer project (Sanger Institute) provides
IC50 dose-response measurements across hundreds of drugs and cell lines.

| File | What it contains |
|------|-----------------|
| `GDSC2_fitted_dose_response_24Jul22.csv` | LN_IC50 for ~200K (cell, drug) pairs, 286 drugs × 967 cell lines |

---

## Where the files come from

The download script fetches all files automatically:

| File | URL |
|------|-----|
| DepMap expression | https://ndownloader.figshare.com/files/51065489 |
| DepMap mutations | https://ndownloader.figshare.com/files/51065732 |
| DepMap CNV | https://ndownloader.figshare.com/files/51065324 |
| DepMap Model.csv | https://ndownloader.figshare.com/files/51065297 |
| CCLE metabolomics | https://data.broadinstitute.org/ccle/CCLE_metabolomics_20190502.csv |
| CCLE RPPA | https://data.broadinstitute.org/ccle/CCLE_RPPA_20181003.csv |
| GDSC2 IC50 | https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/GDSC2_fitted_dose_response_24Jul22.csv |

Total download size: ~1.5 GB. Expect 20–40 min on a typical connection.

---

## What preprocessing does

Raw files are cleaned and saved as per-modality parquet matrices to `data/processed/`.
All rows are cell lines (indexed by DepMap ID); all columns are features.

| Step | Decision | Reason |
|------|----------|--------|
| RNA-seq | Z-score per gene across cell lines | Remove mean expression differences between genes |
| Mutations | Filter synonymous variants; pivot to binary gene×cell matrix; drop genes mutated in <1% of lines | Reduces noise and dimensionality |
| CNV | Fill NaN with per-gene median, then z-score | Sparse missingness in CNV profiles |
| Metabolomics / RPPA | Fill NaN with per-feature median, then z-score | Same rationale |
| Drug response | Map Sanger Model ID → DepMap ID via Model.csv; drop drugs tested in <100 cell lines | 286 drugs pass the threshold |
| Overlap | Intersect all 5 omics + GDSC2 | 597 cell lines have complete data across all modalities |

---

## Outputs

Written to `data/processed/` (git-ignored):

| File | Shape | Description |
|------|-------|-------------|
| `rna.parquet` | 1673 × 19193 | RNA-seq, z-scored |
| `mutations.parquet` | 1929 × 12301 | Binary somatic mutations |
| `cnv.parquet` | 1929 × 38590 | Copy number, z-scored |
| `metabolomics.parquet` | 926 × 225 | Metabolite abundances, z-scored |
| `rppa.parquet` | 894 × 214 | RPPA protein expression, z-scored |
| `drug_response.parquet` | 241,578 × 3 | `(depmap_id, drug_name, ln_ic50)` |
| `cell_line_index.parquet` | 1972 × 3 | `depmap_id ↔ ccle_name ↔ cosmic_id` |
| `overlap_cell_lines.parquet` | 597 × 1 | Cell lines with all 5 omics + GDSC2 |

The `overlap_cell_lines.parquet` file is the primary cell line universe used by
all experiments unless otherwise noted.

---

## Scripts

```
jobs/
  download.py    Fetch all raw files to data/raw/
  preprocess.py  Build data/processed/ from data/raw/
```
