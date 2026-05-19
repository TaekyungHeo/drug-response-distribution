"""Download and process GDSC1 drug response data.

GDSC1 is the companion dataset to GDSC2. It uses the same cell lines (DepMap/Sanger)
but tests a partially different drug set. Merging GDSC1 + GDSC2 increases training
drug count from ~286 to ~400, enabling better drug-blind generalisation.

Output: data/processed/drug_response_gdsc1_gdsc2.parquet
  Combined dataset with a 'source' column ('gdsc1' or 'gdsc2').
  Duplicate (cell_line, drug) pairs: keep the GDSC2 measurement (more recent protocol).
"""

import logging
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

GDSC1_URL = "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC1_fitted_dose_response_24Jul22.csv"
GDSC1_FILENAME = "GDSC1_fitted_dose_response_24Jul22.csv"
GDSC2_FILENAME = "GDSC2_fitted_dose_response_24Jul22.csv"

MIN_CELL_LINES_PER_DRUG = 50  # lower than GDSC2-only (100) to include more drugs

log = logging.getLogger(__name__)


def download_gdsc1(force: bool = False) -> Path:
    """Download GDSC1 CSV to data/raw/ if not already present."""
    dest = RAW_DIR / GDSC1_FILENAME
    if dest.exists() and not force:
        log.info("GDSC1 already downloaded: %s", dest)
        return dest
    log.info("Downloading GDSC1 from %s ...", GDSC1_URL)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(GDSC1_URL, dest)
    log.info("Saved to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def _load_sanger_map() -> pd.Series:
    """Load SangerModelID → DepMap ModelID mapping from Model.csv."""
    model = pd.read_csv(
        RAW_DIR / "Model.csv", usecols=["ModelID", "SangerModelID"], low_memory=False
    )
    model = model.dropna(subset=["SangerModelID"]).drop_duplicates(subset=["SangerModelID"])
    return model.set_index("SangerModelID")["ModelID"]


def _process_gdsc_file(csv_path: Path, sanger_map: pd.Series, source: str) -> pd.DataFrame:
    """Load a GDSC CSV, map to DepMap IDs, return clean (depmap_id, drug_name, ln_ic50, source)."""
    df = pd.read_csv(
        csv_path, usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"], low_memory=False
    )
    df.columns = ["sanger_id", "drug_name", "ln_ic50"]
    df = df.dropna()
    df["depmap_id"] = df["sanger_id"].map(sanger_map)
    df = df.dropna(subset=["depmap_id"])
    df["source"] = source
    log.info(
        "%s: %d pairs, %d drugs, %d cell lines (before overlap filter)",
        source.upper(),
        len(df),
        df["drug_name"].nunique(),
        df["depmap_id"].nunique(),
    )
    return df[["depmap_id", "drug_name", "ln_ic50", "source"]]


def build_combined_drug_response(
    overlap_cell_lines: pd.Index | None = None,
    min_cell_lines: int = MIN_CELL_LINES_PER_DRUG,
) -> pd.DataFrame:
    """Build combined GDSC1+GDSC2 drug response table.

    Args:
        overlap_cell_lines: If provided, filter to these DepMap IDs.
        min_cell_lines: Minimum number of cell lines a drug must be tested on.

    Returns:
        DataFrame with columns (depmap_id, drug_name, ln_ic50, source).
    """
    gdsc1_path = RAW_DIR / GDSC1_FILENAME
    if not gdsc1_path.exists():
        download_gdsc1()

    sanger_map = _load_sanger_map()

    gdsc2 = _process_gdsc_file(RAW_DIR / GDSC2_FILENAME, sanger_map, "gdsc2")
    gdsc1 = _process_gdsc_file(gdsc1_path, sanger_map, "gdsc1")

    # Combine: for duplicate (cell_line, drug) pairs prefer GDSC2 (more recent)
    combined = pd.concat([gdsc2, gdsc1], ignore_index=True)
    combined = combined.drop_duplicates(subset=["depmap_id", "drug_name"], keep="first")

    if overlap_cell_lines is not None:
        combined = combined[combined["depmap_id"].isin(overlap_cell_lines)]

    # Filter drugs tested on at least min_cell_lines
    drug_counts = combined.groupby("drug_name")["depmap_id"].nunique()
    valid_drugs = drug_counts[drug_counts >= min_cell_lines].index
    combined = combined[combined["drug_name"].isin(valid_drugs)]

    log.info(
        "Combined GDSC1+GDSC2: %d pairs, %d drugs, %d cell lines",
        len(combined),
        combined["drug_name"].nunique(),
        combined["depmap_id"].nunique(),
    )
    return combined.reset_index(drop=True)


def get_combined_drug_response(
    overlap_cell_lines: pd.Index | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Load combined drug response from cache or rebuild."""
    out_path = PROCESSED_DIR / "drug_response_combined.parquet"
    if out_path.exists() and not force_recompute:
        log.info("Loading cached combined drug response from %s", out_path)
        return pd.read_parquet(out_path)
    df = build_combined_drug_response(overlap_cell_lines=overlap_cell_lines)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Saved combined drug response to %s", out_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    df = build_combined_drug_response(
        overlap_cell_lines=overlap["depmap_id"],
        min_cell_lines=50,
    )
    print("\nCombined dataset:")
    print(f"  Pairs:      {len(df):,}")
    print(f"  Drugs:      {df['drug_name'].nunique()}")
    print(f"  Cell lines: {df['depmap_id'].nunique()}")
    print(f"  GDSC1 only: {(df['source'] == 'gdsc1').sum():,} pairs")
    print(f"  GDSC2 only: {(df['source'] == 'gdsc2').sum():,} pairs")
    new_drugs = set(df[df["source"] == "gdsc1"]["drug_name"]) - set(
        df[df["source"] == "gdsc2"]["drug_name"]
    )
    print(f"  New drugs from GDSC1: {len(new_drugs)}")
