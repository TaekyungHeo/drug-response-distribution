"""Compute KEGG_MEDICUS pathway-level activity scores for cell lines.

from __future__ import annotations

For each cell line, transforms raw gene expression and mutation data into
619-dimensional pathway activity scores using Mann-Whitney U-statistic
(expression) and mutation fraction (mutations).

Based on PASO (Wu et al., PLoS Comp Bio 2025): pathway-level difference
features improve drug-blind generalization from r≈0.46 to r=0.745.

Output: data/processed/pathway_features.parquet  shape=(n_cells, n_pathways)
        data/processed/pathway_index.json         col_name → pathway metadata
"""

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

GMT_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt"
GMT_FILENAME = "c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt"

MIN_OVERLAP = 3

log = logging.getLogger(__name__)


def download_kegg_medicus_gmt(dest: Path | None = None) -> Path:
    """Download KEGG_MEDICUS GMT file to data/raw/. Skip if already present."""
    if dest is None:
        dest = RAW_DIR / GMT_FILENAME
    if dest.exists():
        log.info("GMT already present: %s", dest)
        return dest
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading KEGG_MEDICUS GMT from MSigDB...")
    urllib.request.urlretrieve(GMT_URL, dest)
    log.info("Saved GMT to %s (%.1f KB)", dest, dest.stat().st_size / 1e3)
    return dest


def parse_gmt(gmt_path: Path) -> dict[str, list[str]]:
    """Parse GMT file into {pathway_name: [gene_symbol, ...]}."""
    pathway_gene_sets: dict[str, list[str]] = {}
    with gmt_path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g for g in parts[2:] if g]
            pathway_gene_sets[name] = genes
    log.info("Parsed %d pathways from GMT", len(pathway_gene_sets))
    return pathway_gene_sets


def compute_rna_pathway_scores(
    rna_df: pd.DataFrame,
    pathway_gene_sets: dict[str, list[str]],
    min_overlap: int = MIN_OVERLAP,
) -> pd.DataFrame:
    """Compute normalized Mann-Whitney U pathway activity scores.

    For each pathway P with k genes found in rna_df:
        rank_sum = sum of expression ranks (ascending) for in-pathway genes
        U = rank_sum - k*(k-1)//2
        score = 2*U / (k * (n_genes - k)) - 1   ∈ [-1, 1]

    Positive score: pathway genes are expressed higher than background.

    Args:
        rna_df: (n_cells, n_genes) DataFrame, index=depmap_id.
        pathway_gene_sets: {pathway_name: [gene_symbol, ...]}.
        min_overlap: Minimum genes a pathway must share with rna_df columns.

    Returns:
        DataFrame (n_cells, n_kept_pathways), index=depmap_id.
    """
    gene_cols = rna_df.columns.tolist()
    gene_to_col: dict[str, int] = {g: i for i, g in enumerate(gene_cols)}
    n_cells, n_genes = rna_df.shape

    log.info("Pre-computing expression ranks for %d cells × %d genes...", n_cells, n_genes)
    rna_arr = rna_df.values.astype(np.float32)
    order = rna_arr.argsort(axis=1)
    global_ranks = np.empty_like(order, dtype=np.int32)
    rows_idx = np.arange(n_cells)[:, None]
    global_ranks[rows_idx, order] = np.arange(n_genes, dtype=np.int32)

    scores: dict[str, np.ndarray] = {}
    skipped = 0
    for name, genes in pathway_gene_sets.items():
        in_idx = [gene_to_col[g] for g in genes if g in gene_to_col]
        k = len(in_idx)
        if k < min_overlap:
            skipped += 1
            continue
        n_out = n_genes - k
        r_in = global_ranks[:, in_idx].sum(axis=1).astype(np.float64)
        u = r_in - k * (k - 1) / 2
        score = 2 * u / (k * n_out) - 1
        scores[name] = score.astype(np.float32)

    log.info(
        "RNA pathway scores: %d kept, %d skipped (<%d gene overlap)",
        len(scores),
        skipped,
        min_overlap,
    )
    return pd.DataFrame(scores, index=rna_df.index)


def compute_mutation_pathway_scores(
    mut_df: pd.DataFrame,
    pathway_gene_sets: dict[str, list[str]],
    min_overlap: int = MIN_OVERLAP,
) -> pd.DataFrame:
    """Compute fraction of pathway genes mutated per cell line.

    Args:
        mut_df: (n_cells, n_genes) binary DataFrame, index=depmap_id.
        pathway_gene_sets: {pathway_name: [gene_symbol, ...]}.
        min_overlap: Minimum genes a pathway must share with mut_df columns.

    Returns:
        DataFrame (n_cells, n_kept_pathways), index=depmap_id.
    """
    gene_cols = set(mut_df.columns)
    mut_arr = mut_df.values.astype(np.float32)
    col_to_idx: dict[str, int] = {g: i for i, g in enumerate(mut_df.columns)}

    scores: dict[str, np.ndarray] = {}
    skipped = 0
    for name, genes in pathway_gene_sets.items():
        in_idx = [col_to_idx[g] for g in genes if g in gene_cols]
        k = len(in_idx)
        if k < min_overlap:
            skipped += 1
            continue
        scores[name] = mut_arr[:, in_idx].mean(axis=1)

    log.info(
        "Mutation pathway scores: %d kept, %d skipped (<%d gene overlap)",
        len(scores),
        skipped,
        min_overlap,
    )
    return pd.DataFrame(scores, index=mut_df.index)


def build_pathway_features(
    rna_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    pathway_gene_sets: dict[str, list[str]],
    min_overlap: int = MIN_OVERLAP,
    include_mutation: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build full pathway feature matrix combining RNA scores and mutation fractions.

    Columns are named {pathway}_rna and {pathway}_mut (if include_mutation=True).
    Output is aligned to rna_df.index; cells missing from mut_df get zero mutation scores.

    Returns:
        features_df: (n_cells, n_features) DataFrame, index=depmap_id, float32.
        metadata: dict with coverage stats and pathway info.
    """
    rna_scores = compute_rna_pathway_scores(rna_df, pathway_gene_sets, min_overlap)
    rna_scores.columns = [f"{c}_rna" for c in rna_scores.columns]

    if include_mutation:
        mut_scores = compute_mutation_pathway_scores(mut_df, pathway_gene_sets, min_overlap)
        mut_scores.columns = [f"{c}_mut" for c in mut_scores.columns]
        # Align mutation scores to rna_df index; fill missing with 0
        mut_aligned = mut_scores.reindex(rna_df.index, fill_value=0.0)
        features_df = pd.concat([rna_scores, mut_aligned], axis=1)
    else:
        features_df = rna_scores

    n_rna_pathways = rna_scores.shape[1]
    n_mut_pathways = mut_scores.shape[1] if include_mutation else 0
    log.info(
        "Combined pathway features: %d cells × %d features (%d RNA + %d mut)",
        len(features_df),
        features_df.shape[1],
        n_rna_pathways,
        n_mut_pathways,
    )

    metadata: dict[str, Any] = {
        "n_cells": len(features_df),
        "n_features": features_df.shape[1],
        "n_rna_pathways": n_rna_pathways,
        "n_mut_pathways": n_mut_pathways,
        "include_mutation": include_mutation,
        "min_overlap": min_overlap,
    }
    return features_df.astype(np.float32), metadata


def get_pathway_features(
    processed_dir: Path = PROCESSED_DIR,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Load pathway features from cache or compute and save them."""
    feat_path = processed_dir / "pathway_features.parquet"
    if feat_path.exists() and not force_recompute:
        log.info("Loading cached pathway features from %s", feat_path)
        return pd.read_parquet(feat_path)

    gmt_path = download_kegg_medicus_gmt()
    pathway_gene_sets = parse_gmt(gmt_path)

    rna_df = pd.read_parquet(processed_dir / "rna.parquet")
    mut_df = pd.read_parquet(processed_dir / "mutations.parquet")

    features_df, metadata = build_pathway_features(rna_df, mut_df, pathway_gene_sets)

    processed_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(feat_path)
    idx_path = processed_dir / "pathway_index.json"
    with idx_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved pathway features %s to %s", features_df.shape, feat_path)
    return features_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = get_pathway_features(force_recompute=True)
    print(f"Shape: {df.shape}")
    print(f"Cells: {len(df)}")
    print(f"Sample columns: {list(df.columns[:3])}")
    print(f"Value range: [{df.values.min():.3f}, {df.values.max():.3f}]")
    print(f"Non-zero fraction: {(df.values != 0).mean():.3f}")
