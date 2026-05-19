"""CTRPv2 dataset loading and preprocessing.

CTRPv2 (Cancer Therapeutics Response Portal v2): ~900 cell lines, 545 drugs, AUC.
Raw files live in data/raw/ctrpv2/extracted/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_RAW_DIR = Path("data/raw/ctrpv2/extracted")
_DEFAULT_PROCESSED_DIR = Path("data/processed")
_MIN_CELLS_PER_DRUG = 5


def load_ctrpv2_response(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Load CTRPv2 drug response and map to DepMap cell IDs.

    Returns:
        DataFrame with columns: depmap_id, drug_name, auc
    """
    if raw_dir is None:
        raw_dir = Path(_DEFAULT_RAW_DIR)
    if processed_dir is None:
        processed_dir = Path(_DEFAULT_PROCESSED_DIR)

    curves = pd.read_csv(
        raw_dir / "v20.data.curves_post_qc.txt", sep="\t",
        usecols=["experiment_id", "area_under_curve", "master_cpd_id"],
    )
    exp_meta = pd.read_csv(
        raw_dir / "v20.meta.per_experiment.txt", sep="\t",
        usecols=["experiment_id", "master_ccl_id"],
    )
    cell_meta = pd.read_csv(
        raw_dir / "v20.meta.per_cell_line.txt", sep="\t",
        usecols=["master_ccl_id", "ccl_name"],
    )
    cpd_meta = pd.read_csv(
        raw_dir / "v20.meta.per_compound.txt", sep="\t",
        usecols=["master_cpd_id", "cpd_name"],
    )

    df = (
        curves
        .merge(exp_meta, on="experiment_id")
        .merge(cell_meta, on="master_ccl_id")
        .merge(cpd_meta, on="master_cpd_id")
    )
    df = df[["ccl_name", "cpd_name", "area_under_curve"]].dropna()
    df["cpd_name"] = df["cpd_name"].str.strip()
    df["ccl_name"] = df["ccl_name"].str.strip()
    df = df.groupby(["ccl_name", "cpd_name"])["area_under_curve"].mean().reset_index()

    # Map CTRP cell names → DepMap IDs using cell_line_index
    cl_idx = pd.read_parquet(processed_dir / "cell_line_index.parquet")
    name_to_depmap: dict[str, str] = {
        str(row["stripped_name"]).upper().replace("-", "").replace(" ", "").replace("_", ""): str(dep)
        for dep, row in cl_idx.iterrows()
    }

    def _match(name: str) -> str | None:
        return name_to_depmap.get(
            str(name).upper().replace("-", "").replace(" ", "").replace("_", "")
        )

    df["depmap_id"] = df["ccl_name"].map(_match)
    df = df.dropna(subset=["depmap_id"]).copy()
    df = df.rename(columns={"cpd_name": "drug_name", "area_under_curve": "auc"})
    df = df[["depmap_id", "drug_name", "auc"]].reset_index(drop=True)

    logger.info(
        "CTRPv2: %d drugs, %d cells, %d pairs",
        df["drug_name"].nunique(), df["depmap_id"].nunique(), len(df),
    )
    return df


def filter_ctrpv2(
    df: pd.DataFrame,
    rna_index: pd.Index,
    mut_index: pd.Index,
    min_cells: int = _MIN_CELLS_PER_DRUG,
) -> pd.DataFrame:
    """Restrict to cells with omics data and drugs with sufficient coverage.

    Args:
        df: Output of load_ctrpv2_response().
        rna_index: Index of cells with RNA features.
        mut_index: Index of cells with mutation features.
        min_cells: Minimum cells per drug.

    Returns:
        Filtered DataFrame.
    """
    valid_cells = set(rna_index) & set(mut_index)
    df = df[df["depmap_id"].isin(valid_cells)].copy()
    counts = df.groupby("drug_name")["depmap_id"].nunique()
    valid = counts[counts >= min_cells].index
    df = df[df["drug_name"].isin(valid)].reset_index(drop=True)
    logger.info(
        "CTRPv2 after filter: %d drugs, %d cells",
        df["drug_name"].nunique(), df["depmap_id"].nunique(),
    )
    return df


__all__ = ["filter_ctrpv2", "load_ctrpv2_response"]
