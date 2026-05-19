"""Drug Repurposing Hub MoA annotation loading.

Covers ~6,800 compounds with curated mechanism-of-action (MoA) and target
annotations. PRISM Repurposing Screen drugs are a subset of this library
(~97% coverage). BeatAML kinase-inhibitor drugs have ~45% coverage.

Source: https://repo-hub.broadinstitute.org/repurposing (CC-BY 4.0)
Local file: data/processed/repurposing_hub_moa.tsv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/processed/repurposing_hub_moa.tsv")


def load_repurposing_hub(path: Path | None = None) -> pd.DataFrame:
    """Load the Repurposing Hub annotation table.

    Returns:
        DataFrame with columns: pert_iname, moa, target, clinical_phase
    """
    if path is None:
        path = Path(_DEFAULT_PATH)
    df = pd.read_csv(path, sep="\t", comment="!")
    df = df[["pert_iname", "moa", "target", "clinical_phase"]].copy()
    logger.info("Repurposing Hub: %d compounds, %d with MoA", len(df), df["moa"].notna().sum())
    return df


def build_drug_moa_map(
    drug_names: list[str],
    hub: pd.DataFrame | None = None,
    path: Path | None = None,
    top_moa_only: bool = True,
) -> dict[str, str | None]:
    """Map drug names to their primary MoA string from the Repurposing Hub.

    Matching is case-insensitive and strips whitespace.

    Args:
        drug_names: Drug names to annotate.
        hub: Pre-loaded Repurposing Hub DataFrame (optional).
        path: Path to TSV file (used if hub is None).
        top_moa_only: If True and MoA is a pipe-separated list, return only
                      the first entry.

    Returns:
        Dict mapping each drug name to its MoA string (or None if not found).
    """
    if hub is None:
        hub = load_repurposing_hub(path)

    index: dict[str, str | None] = {
        str(row["pert_iname"]).lower().strip(): (
            str(row["moa"]).split("|")[0].strip() if top_moa_only else str(row["moa"])
        ) if pd.notna(row["moa"]) else None
        for _, row in hub.iterrows()
    }

    result: dict[str, str | None] = {}
    for name in drug_names:
        result[name] = index.get(str(name).lower().strip())
    n_matched = sum(v is not None for v in result.values())
    logger.info(
        "MoA lookup: %d/%d drugs matched (%.0f%%)",
        n_matched, len(drug_names), 100 * n_matched / max(len(drug_names), 1),
    )
    return result


def group_by_moa(
    drug_moa: dict[str, str | None],
    min_drugs: int = 3,
) -> dict[str, list[str]]:
    """Group drugs by MoA class, keeping only classes with enough drugs.

    Args:
        drug_moa: Output of build_drug_moa_map().
        min_drugs: Minimum number of drugs to keep a MoA class.

    Returns:
        Dict mapping MoA label → list of drug names.
    """
    groups: dict[str, list[str]] = {}
    for drug, moa in drug_moa.items():
        if moa is None:
            continue
        groups.setdefault(moa, []).append(drug)
    return {k: v for k, v in groups.items() if len(v) >= min_drugs}


__all__ = ["build_drug_moa_map", "group_by_moa", "load_repurposing_hub"]
