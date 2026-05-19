"""PRISM Repurposing dataset loading and preprocessing.

Shared by 05_dataset_robustness ridge and transformer scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_prism(data_dir: Path) -> pd.DataFrame:
    """Load PRISM drug response data.

    Tries processed parquet files first; falls back to raw DepMap CSV.

    Returns:
        DataFrame with columns: depmap_id, drug_name, response (float)
    """
    processed_path = data_dir / "prism_drug_response.parquet"
    repurposing_path = data_dir / "prism_repurposing.parquet"

    if processed_path.exists():
        df = pd.read_parquet(processed_path)
        logger.info("Loaded PRISM from %s: %s", processed_path.name, df.shape)
        if "ln_ic50" in df.columns:
            return df.rename(columns={"ln_ic50": "response"})
        elif "auc" in df.columns:
            return df.rename(columns={"auc": "response"})
        else:
            raise ValueError(
                f"Unrecognised response column in {processed_path}: {df.columns.tolist()}"
            )

    if repurposing_path.exists():
        df = pd.read_parquet(repurposing_path)
        logger.info("Loaded PRISM from %s: %s", repurposing_path.name, df.shape)
        if "auc" in df.columns:
            df = df.rename(columns={"auc": "response"})
            if "broad_id" in df.columns:
                df = df.rename(columns={"broad_id": "drug_name"})
        elif "ln_ic50" in df.columns:
            df = df.rename(columns={"ln_ic50": "response"})
        else:
            raise ValueError(
                f"Unrecognised columns in {repurposing_path}: {df.columns.tolist()}"
            )
        return df

    # Raw CSV fallback (DepMap portal download)
    csv_path = data_dir.parent / "external" / "prism" / "repurposing_secondary_screen_dose_response.csv"
    if csv_path.exists():
        logger.info("Processing raw PRISM CSV (this may take a moment)...")
        raw = pd.read_csv(csv_path, low_memory=False)
        logger.info("Raw CSV shape: %s  columns: %s", raw.shape, raw.columns.tolist())
        col_map: dict[str, str] = {}
        for col in raw.columns:
            if col.lower() in ("broad_id", "column_name"):
                col_map[col] = "drug_name"
            elif col.lower() == "depmap_id":
                col_map[col] = "depmap_id"
            elif col.lower() == "auc":
                col_map[col] = "response"
        if "drug_name" not in col_map.values() or "response" not in col_map.values():
            raise ValueError(
                f"Cannot map required columns from raw CSV. Found: {raw.columns.tolist()}"
            )
        df = pd.DataFrame(
            raw.rename(columns=col_map)[["depmap_id", "drug_name", "response"]].dropna(
                subset=["depmap_id", "drug_name"]  # type: ignore[call-overload]
            )
        )
        df_save = df.rename(columns={"response": "auc"})
        df_save.to_parquet(repurposing_path, index=False)
        logger.info("Saved processed parquet to %s", repurposing_path)
        return df

    raise FileNotFoundError(
        "PRISM data not found. Expected one of:\n"
        f"  {processed_path}\n"
        f"  {repurposing_path}\n"
        f"  {csv_path}\n"
        "Download repurposing_secondary_screen_dose_response.csv from the DepMap portal."
    )


def preprocess_prism(
    df: pd.DataFrame,
    rna_cell_ids: set,
    min_cells_per_drug: int = 50,
) -> tuple[pd.DataFrame, int, int]:
    """Filter PRISM to usable drugs and cells.

    Steps:
    1. Exclude drugs with > 10% failed wells (NaN response).
    2. Intersect cell lines with rna_cell_ids.
    3. Exclude drugs with < min_cells_per_drug cell lines after intersection.

    Returns:
        (filtered_df, n_drugs, n_cells)
    """
    total_drugs = df["drug_name"].nunique()
    total_cells = df["depmap_id"].nunique()
    logger.info("PRISM raw: %d drugs, %d cells, %d pairs", total_drugs, total_cells, len(df))

    drug_total = df.groupby("drug_name").size()
    drug_nan = df.groupby("drug_name")["response"].apply(lambda x: x.isna().sum())
    high_fail = (drug_nan / drug_total)[lambda x: x > 0.10].index
    if len(high_fail) > 0:
        logger.warning("Excluding %d drugs with >10%% failed wells", len(high_fail))
        df = df[~df["drug_name"].isin(high_fail)].copy()

    df = pd.DataFrame(df.dropna(subset=["response"]).reset_index(drop=True))
    df = pd.DataFrame(df[df["depmap_id"].isin(rna_cell_ids)].reset_index(drop=True))

    per_drug = df.groupby("drug_name").size()
    valid_drugs = per_drug[per_drug >= min_cells_per_drug].index  # type: ignore[union-attr]
    df = pd.DataFrame(df[df["drug_name"].isin(valid_drugs)].reset_index(drop=True))

    n_drugs = df["drug_name"].nunique()
    n_cells = df["depmap_id"].nunique()
    logger.info("After filtering: %d drugs, %d cells, %d pairs", n_drugs, n_cells, len(df))
    return df, n_drugs, n_cells


__all__ = ["load_prism", "preprocess_prism"]
