"""MultiOmicsDataset variant that loads from a combined drug response table.

from __future__ import annotations

Identical to MultiOmicsDataset but accepts an external drug_response DataFrame,
enabling multi-dataset training (GDSC1 + GDSC2) without modifying processed files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ['CombinedDataset']

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MODALITIES = ["rna", "mutations", "cnv", "metabolomics", "rppa"]


class CombinedDataset(Dataset):
    """Dataset backed by an arbitrary drug_response DataFrame.

    Args:
        drug_response: DataFrame with columns (depmap_id, drug_name, ln_ic50).
        cell_lines: DepMap IDs to include (rows in drug_response must match).
        omics_to_use: Which modalities to load.
        modality_dropout_p: Per-modality dropout probability during training.
        processed_dir: Path to processed omics parquet files.
    """

    def __init__(
        self,
        drug_response: pd.DataFrame,
        cell_lines: list[str] | None = None,
        omics_to_use: list[str] | None = None,
        modality_dropout_p: float = 0.0,
        processed_dir: Path = PROCESSED_DIR,
    ) -> None:
        self.modality_dropout_p = modality_dropout_p
        self.omics_to_use = omics_to_use or MODALITIES

        drug_df = drug_response.copy()
        if cell_lines is not None:
            drug_df = drug_df[drug_df["depmap_id"].isin(cell_lines)]
        drug_df = drug_df.reset_index(drop=True)
        self.pairs = drug_df

        drugs = sorted(drug_df["drug_name"].unique())
        self.drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(drugs)}
        self.n_drugs = len(drugs)

        unique_cells = sorted(drug_df["depmap_id"].unique())
        cell_to_row: dict[str, int] = {c: i for i, c in enumerate(unique_cells)}
        self.n_cell_lines = len(unique_cells)

        self.omics_arrays: dict[str, np.ndarray] = {}
        self.feature_dims: dict[str, int] = {}

        for mod in self.omics_to_use:
            path = processed_dir / f"{mod}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Processed file not found: {path}")
            df = pd.read_parquet(path)
            n_feat = df.shape[1]
            self.feature_dims[mod] = n_feat
            arr = np.zeros((self.n_cell_lines, n_feat), dtype=np.float32)
            for cell_id, row_idx in cell_to_row.items():
                if cell_id in df.index:
                    arr[row_idx] = df.loc[cell_id].values.astype(np.float32)
            self.omics_arrays[mod] = arr

        self._cell_rows = np.array([cell_to_row[c] for c in drug_df["depmap_id"]], dtype=np.int64)
        self._drug_idxs = np.array(
            [self.drug_to_idx[d] for d in drug_df["drug_name"]], dtype=np.int64
        )
        self._targets = drug_df["ln_ic50"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> tuple[dict[str, Tensor], int, float]:
        cell_row = int(self._cell_rows[idx])
        drug_idx = int(self._drug_idxs[idx])
        ln_ic50 = float(self._targets[idx])
        omics_tensors: dict[str, Tensor] = {}
        for mod in self.omics_to_use:
            values = self.omics_arrays[mod][cell_row].copy()
            if self.modality_dropout_p > 0.0 and np.random.random() < self.modality_dropout_p:
                values[:] = 0.0
            omics_tensors[mod] = torch.from_numpy(values)
        return omics_tensors, drug_idx, ln_ic50

    @property
    def concat_dim(self) -> int:
        return sum(self.feature_dims.values())

    def get_targets(self, idx: np.ndarray) -> np.ndarray:
        return self._targets[idx]
