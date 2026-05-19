"""PyTorch Dataset for (cell_line, drug) multi-omics pairs.

from __future__ import annotations

Optimized for throughput:
- All omics loaded into numpy arrays at init; __getitem__ is pure numpy indexing.
- Pairs stored as pre-encoded integer arrays (depmap_idx, drug_idx, ln_ic50).
- No pandas overhead at __getitem__ time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ['MultiOmicsDataset']

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

MODALITIES = ["rna", "mutations", "cnv", "metabolomics", "rppa"]


class MultiOmicsDataset(Dataset):
    """Dataset of (cell_line, drug) pairs with multi-omics features and LN_IC50 targets.

    Args:
        cell_lines: DepMap IDs to include (must be in processed data).
        omics_to_use: Which modalities to load. Defaults to all five.
        modality_dropout_p: Probability of dropping each modality per sample during training.
            Set to 0.0 at inference time.
    """

    def __init__(
        self,
        cell_lines: list[str] | None = None,
        omics_to_use: list[str] | None = None,
        modality_dropout_p: float = 0.0,
        drug_response_file: str = "drug_response.parquet",
    ) -> None:
        self.modality_dropout_p = modality_dropout_p
        self.omics_to_use = omics_to_use or MODALITIES

        # Load drug response pairs
        drug_df = pd.read_parquet(PROCESSED_DIR / drug_response_file)
        if cell_lines is not None:
            drug_df = drug_df[drug_df["depmap_id"].isin(cell_lines)]
        drug_df = drug_df.reset_index(drop=True)
        self.pairs = drug_df  # kept for split utilities that need depmap_id/drug_name

        # Encode drug names as integer IDs
        drugs = sorted(drug_df["drug_name"].unique())
        self.drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(drugs)}
        self.n_drugs = len(drugs)

        # --- Build cell-line integer index ---
        # All unique depmap_ids that appear in pairs
        unique_cells = sorted(drug_df["depmap_id"].unique())
        cell_to_row: dict[str, int] = {c: i for i, c in enumerate(unique_cells)}
        self.n_cell_lines = len(unique_cells)

        # Load omics matrices as float32 numpy arrays (rows = cell lines)
        # Rows are aligned to unique_cells order; missing rows → zeros.
        self.omics_arrays: dict[str, np.ndarray] = {}
        self.feature_dims: dict[str, int] = {}

        for mod in self.omics_to_use:
            path = PROCESSED_DIR / f"{mod}.parquet"
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

        # Pre-encode pairs as integer arrays for zero-overhead __getitem__
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
        """Fast batch target retrieval — avoids Python loop over __getitem__."""
        return self._targets[idx]

    # ------------------------------------------------------------------
    # Public accessors for internal arrays (encapsulation boundary)
    # ------------------------------------------------------------------

    def to_concat_array(self) -> np.ndarray:
        """Return the (n_cells, total_omics_features) concatenated feature matrix.

        Rows are aligned to the internal cell ordering. Used by training loops
        to avoid repeated concatenation. The caller should not modify the result.
        """
        arrays = [self.omics_arrays[mod] for mod in self.omics_to_use]
        return np.ascontiguousarray(np.concatenate(arrays, axis=1), dtype=np.float32)

    @property
    def cell_rows(self) -> np.ndarray:
        """Integer row indices into to_concat_array() for each pair."""
        return self._cell_rows

    @property
    def drug_indices(self) -> np.ndarray:
        """Integer drug index for each pair (aligned to drug_to_idx)."""
        return self._drug_idxs

    @property
    def targets(self) -> np.ndarray:
        """LN_IC50 target values for each pair."""
        return self._targets
