"""Preprocess raw CCLE/DepMap and GDSC files into clean per-modality parquet files.

Must be run after download.py. Reads from data/raw/, writes to data/processed/.

Outputs:
  data/processed/
    rna.parquet             rows=cell_lines, cols=genes (z-scored)
    mutations.parquet       rows=cell_lines, cols=genes (binary, ≥1% prevalence)
    cnv.parquet             rows=cell_lines, cols=genes (z-scored, NaN→median)
    metabolomics.parquet    rows=cell_lines, cols=metabolites (z-scored)
    rppa.parquet            rows=cell_lines, cols=proteins (z-scored)
    drug_response.parquet   (depmap_id, drug_name, ln_ic50) — drugs ≥100 cell lines
    cell_line_index.parquet depmap_id ↔ ccle_name ↔ cosmic_id mapping
    overlap_cell_lines.parquet  DepMap IDs with all 5 omics + GDSC2 response

Usage:
    uv run python3 experiments/00_data_preparation/jobs/preprocess.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardize each column (feature) across cell lines."""
    return (df - df.mean()) / (df.std() + 1e-8)


def _save(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path)
    logger.info("  saved  %-40s shape=%s", path.name, df.shape)


# ---------------------------------------------------------------------------
# Cell line index
# ---------------------------------------------------------------------------

def build_cell_line_index() -> pd.DataFrame:
    """Map DepMap IDs to CCLE names and COSMIC IDs (from Model.csv)."""
    model = pd.read_csv(RAW_DIR / "Model.csv", low_memory=False)
    col_map = {
        "ModelID":              "depmap_id",
        "CCLEName":             "ccle_name",
        "COSMICID":             "cosmic_id",
        "StrippedCellLineName": "stripped_name",
    }
    keep = [k for k in col_map if k in model.columns]
    index = model[keep].copy()
    index.columns = [col_map[k] for k in keep]
    index = index.dropna(subset=["depmap_id"]).set_index("depmap_id")
    _save(index, "cell_line_index")
    return index


# ---------------------------------------------------------------------------
# Omics modalities
# ---------------------------------------------------------------------------

def process_rna(index: pd.DataFrame) -> pd.DataFrame:
    logger.info("RNA-seq...")
    rna = pd.read_csv(RAW_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    rna.columns = [c.split(" ")[0] for c in rna.columns]   # "GENE (ID)" → "GENE"
    rna = rna[rna.index.isin(index.index)]
    rna = _standardize(rna)
    _save(rna, "rna")
    return rna


def process_mutations(index: pd.DataFrame) -> pd.DataFrame:
    logger.info("Somatic mutations...")
    mut = pd.read_csv(RAW_DIR / "OmicsSomaticMutations.csv", low_memory=False)

    # Filter out synonymous variants (column name differs across releases)
    if "VariantType" in mut.columns:
        mut = mut[mut["VariantType"] != "SNP_SYNONYMOUS"]
    elif "VariantClassification" in mut.columns:
        synonymous = {"Silent", "3'UTR", "5'UTR", "Intron", "IGR"}
        mut = mut[~mut["VariantClassification"].isin(synonymous)]

    model_col = next((c for c in mut.columns if "ModelID" in c or "DepMap_ID" in c), None)
    gene_col  = next((c for c in mut.columns if c in ("HugoSymbol", "Gene", "Hugo_Symbol")), None)
    if model_col is None or gene_col is None:
        raise ValueError(f"Cannot find ID/gene columns in mutations. Cols: {mut.columns.tolist()}")

    mut["_val"] = 1
    pivot = mut.pivot_table(
        index=model_col, columns=gene_col, values="_val", aggfunc="max", fill_value=0
    ).astype(np.int8)
    pivot = pivot[pivot.index.isin(index.index)]

    # Keep genes mutated in ≥1% of cell lines
    min_lines = max(1, int(0.01 * len(pivot)))
    pivot = pivot.loc[:, pivot.sum() >= min_lines]
    _save(pivot, "mutations")
    return pivot


def process_cnv(index: pd.DataFrame) -> pd.DataFrame:
    logger.info("Copy number variation...")
    cnv = pd.read_csv(RAW_DIR / "OmicsCNGene.csv", index_col=0)
    cnv.columns = [c.split(" ")[0] for c in cnv.columns]
    cnv = cnv[cnv.index.isin(index.index)]
    cnv = cnv.apply(lambda col: col.fillna(col.median()))
    cnv = _standardize(cnv)
    _save(cnv, "cnv")
    return cnv


def process_metabolomics(index: pd.DataFrame) -> pd.DataFrame:
    logger.info("Metabolomics...")
    met = pd.read_csv(RAW_DIR / "CCLE_metabolomics_20190502.csv")
    met = met.set_index("DepMap_ID").drop(columns=["CCLE_ID"], errors="ignore")
    met = met[met.index.isin(index.index)]
    met = met.apply(lambda col: col.fillna(col.median()))
    met = _standardize(met)
    _save(met, "metabolomics")
    return met


def process_rppa(index: pd.DataFrame) -> pd.DataFrame:
    logger.info("RPPA...")
    rppa = pd.read_csv(RAW_DIR / "CCLE_RPPA_20181003.csv")
    rppa = rppa.rename(columns={"Unnamed: 0": "ccle_name"}).set_index("ccle_name")

    if "ccle_name" in index.columns:
        ccle_to_depmap = (
            index.reset_index().dropna(subset=["ccle_name"]).set_index("ccle_name")["depmap_id"]
        )
        rppa.index = rppa.index.map(ccle_to_depmap)
        rppa = rppa[rppa.index.notna()]

    rppa = rppa[rppa.index.isin(index.index)]
    rppa = rppa.apply(lambda col: col.fillna(col.median()))
    rppa = _standardize(rppa)
    _save(rppa, "rppa")
    return rppa


# ---------------------------------------------------------------------------
# Drug response (GDSC2)
# ---------------------------------------------------------------------------

def process_drug_response(index: pd.DataFrame) -> pd.DataFrame:
    logger.info("GDSC2 drug response...")
    gdsc = pd.read_csv(RAW_DIR / "GDSC2_fitted_dose_response_24Jul22.csv", low_memory=False)
    gdsc = gdsc[["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"]].copy()
    gdsc.columns = ["sanger_id", "drug_name", "ln_ic50"]
    gdsc = gdsc.dropna()

    # Map Sanger model ID → DepMap ID
    if "sanger_id" in index.columns:
        sanger_map = (
            index.reset_index().dropna(subset=["sanger_id"]).set_index("sanger_id")["depmap_id"]
        )
    else:
        model = pd.read_csv(
            RAW_DIR / "Model.csv", usecols=["ModelID", "SangerModelID"], low_memory=False
        )
        model = model.dropna(subset=["SangerModelID"]).drop_duplicates(subset=["SangerModelID"])
        sanger_map = model.set_index("SangerModelID")["ModelID"]

    gdsc["depmap_id"] = gdsc["sanger_id"].map(sanger_map)
    gdsc = gdsc.dropna(subset=["depmap_id"])[["depmap_id", "drug_name", "ln_ic50"]]

    # Keep drugs tested in ≥100 cell lines
    valid_drugs = gdsc.groupby("drug_name")["depmap_id"].nunique()
    valid_drugs = valid_drugs[valid_drugs >= 100].index
    gdsc = gdsc[gdsc["drug_name"].isin(valid_drugs)].reset_index(drop=True)
    _save(gdsc, "drug_response")
    return gdsc


# ---------------------------------------------------------------------------
# Overlap index
# ---------------------------------------------------------------------------

def compute_overlap_index(*dfs: pd.DataFrame) -> pd.Index:
    common = dfs[0].index
    for df in dfs[1:]:
        common = common.intersection(df.index)
    return common


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    missing = [f for f in [
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "OmicsSomaticMutations.csv",
        "OmicsCNGene.csv",
        "Model.csv",
        "CCLE_metabolomics_20190502.csv",
        "CCLE_RPPA_20181003.csv",
        "GDSC2_fitted_dose_response_24Jul22.csv",
    ] if not (RAW_DIR / f).exists()]
    if missing:
        logger.error("Missing raw files in %s:", RAW_DIR)
        for f in missing:
            logger.error("  %s", f)
        logger.error("Run download.py first.")
        sys.exit(1)

    logger.info("=== Preprocessing raw data → %s ===\n", PROCESSED_DIR)

    index = build_cell_line_index()
    logger.info("  Cell line index: %d entries\n", len(index))

    rna  = process_rna(index)
    mut  = process_mutations(index)
    cnv  = process_cnv(index)
    met  = process_metabolomics(index)
    rppa = process_rppa(index)
    drug = process_drug_response(index)

    overlap = compute_overlap_index(rna, mut, cnv, met, rppa)
    overlap_with_drug = overlap.intersection(drug["depmap_id"].unique())

    overlap_df = pd.DataFrame({"depmap_id": overlap_with_drug})
    overlap_df.to_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet", index=False)

    logger.info("\n=== Summary ===")
    logger.info("  RNA:           %d cell lines  %d genes",      len(rna),  rna.shape[1])
    logger.info("  Mutations:     %d cell lines  %d genes",      len(mut),  mut.shape[1])
    logger.info("  CNV:           %d cell lines  %d genes",      len(cnv),  cnv.shape[1])
    logger.info("  Metabolomics:  %d cell lines  %d metabolites", len(met),  met.shape[1])
    logger.info("  RPPA:          %d cell lines  %d proteins",   len(rppa), rppa.shape[1])
    logger.info("  Drug response: %d pairs  %d drugs  %d cells",
                len(drug), drug["drug_name"].nunique(), drug["depmap_id"].nunique())
    logger.info("  Overlap (all 5 omics + GDSC2): %d cell lines", len(overlap_with_drug))
    logger.info("\n  saved  overlap_cell_lines.parquet  (%d entries)", len(overlap_with_drug))


if __name__ == "__main__":
    run()
