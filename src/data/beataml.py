"""BeatAML dataset loading and preprocessing.

BeatAML2 (dbGaP phs001657): 520 patients, 155 drugs, AUC response.
Patient IDs are dbGaP RNA-seq sample IDs (dbgap_rnaseq_sample).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path("data/external/beataml2")
_RESPONSE_FILE = "beataml_probit_curve_fits_v4_dbgap.txt"
_EXPR_FILE = "beataml_waves1to4_norm_exp_dbgap.txt"
_MIN_PATIENTS_PER_DRUG = 20
_TOP_GENES = 5000
_RNA_PCA_DIMS = 500


def load_beataml_response(
    data_dir: Path | None = None,
    min_patients: int = _MIN_PATIENTS_PER_DRUG,
) -> pd.DataFrame:
    """Load BeatAML drug response, filtered to patients with RNA-seq.

    Returns:
        DataFrame with columns: patient_id, drug, auc
    """
    if data_dir is None:
        data_dir = Path(_DEFAULT_DATA_DIR)

    dr = pd.read_csv(data_dir / _RESPONSE_FILE, sep="\t",
                     usecols=["dbgap_rnaseq_sample", "inhibitor", "auc"])
    dr = dr[dr["dbgap_rnaseq_sample"].notna() & (dr["dbgap_rnaseq_sample"] != "")]
    dr = dr[dr["auc"].notna()]

    # Keep only patients that have RNA-seq data
    header = subprocess.run(
        ["head", "-1", str(data_dir / _EXPR_FILE)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split("\t")
    rna_patients = set(header[4:])
    dr = dr[dr["dbgap_rnaseq_sample"].isin(rna_patients)]

    # Filter drugs by coverage
    counts = dr.groupby("inhibitor")["dbgap_rnaseq_sample"].nunique()
    valid = counts[counts >= min_patients].index
    dr = dr[dr["inhibitor"].isin(valid)].copy()

    dr = dr.rename(columns={"dbgap_rnaseq_sample": "patient_id", "inhibitor": "drug"})
    dr = dr[["patient_id", "drug", "auc"]].reset_index(drop=True)
    logger.info(
        "BeatAML response: %d drugs, %d patients, %d pairs",
        dr["drug"].nunique(), dr["patient_id"].nunique(), len(dr),
    )
    return dr


def load_beataml_expression(
    patients: list[str],
    data_dir: Path | None = None,
    top_genes: int = _TOP_GENES,
) -> pd.DataFrame:
    """Load BeatAML RNA-seq expression matrix (protein-coding, variance-filtered).

    Args:
        patients: List of patient IDs to include (rows in returned matrix).
        data_dir: Path to BeatAML data directory.
        top_genes: Keep this many highest-variance genes.

    Returns:
        DataFrame (n_patients × top_genes), index = patient_id.
    """
    if data_dir is None:
        data_dir = Path(_DEFAULT_DATA_DIR)

    patients_set = set(patients)
    expr = pd.read_csv(data_dir / _EXPR_FILE, sep="\t", index_col="stable_id")
    expr = expr[expr["biotype"] == "protein_coding"].drop(
        columns=["display_label", "description", "biotype"], errors="ignore"
    )
    # Keep only requested patients that are present
    cols = [c for c in expr.columns if c in patients_set]
    expr = expr[cols]

    var = expr.T.var(axis=0)
    top = var.nlargest(top_genes).index
    mat = expr.loc[top, cols].T  # (n_patients, top_genes)
    mat.index.name = "patient_id"
    logger.info("BeatAML expression: %d patients × %d genes", *mat.shape)
    return mat


def build_beataml_features(
    response: pd.DataFrame,
    rna_pca_dims: int = _RNA_PCA_DIMS,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load expression, run PCA, return arrays ready for Ridge CV.

    Args:
        response: DataFrame from load_beataml_response().
        rna_pca_dims: Number of PCA dimensions for RNA features.

    Returns:
        (response_filtered, expr_pca, X_cell, y, patients, drugs)
        where X_cell is (n_pairs, rna_pca_dims) and y is (n_pairs,).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    patients = sorted(response["patient_id"].unique())
    expr_raw = load_beataml_expression(patients)

    # Re-align patients to those with both response and expression
    common = sorted(set(patients) & set(expr_raw.index))
    response = response[response["patient_id"].isin(common)].copy()
    expr_raw = expr_raw.loc[common]

    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_raw.values)
    pca = PCA(n_components=min(rna_pca_dims, expr_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(expr_scaled)
    patient_to_row = {p: i for i, p in enumerate(common)}

    drugs = sorted(response["drug"].unique())
    rows, y = [], []
    for _, r in response.iterrows():
        rows.append(patient_to_row[r["patient_id"]])
        y.append(r["auc"])
    X_cell = X_pca[rows]

    logger.info(
        "BeatAML features: %d patients, %d drugs, %d pairs, RNA PCA(%d)",
        len(common), len(drugs), len(y), rna_pca_dims,
    )
    return response, expr_raw, X_cell, np.array(y, dtype=np.float32), common, drugs


__all__ = [
    "build_beataml_features",
    "load_beataml_expression",
    "load_beataml_response",
]
