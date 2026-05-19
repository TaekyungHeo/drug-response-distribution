"""Tests for MultiOmicsDataset public API.

Validates to_concat_array(), cell_rows, drug_indices, and targets properties
without accessing private attributes. Uses a mock dataset built with synthetic
in-memory data so no real data files are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.dataset import MultiOmicsDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_dataset(n_cells: int = 4, n_pairs: int = 8) -> tuple[MultiOmicsDataset, int, int]:
    """Build a MultiOmicsDataset with fake in-memory data (no disk I/O).

    Patches pd.read_parquet so the class constructor never touches the filesystem.
    Returns (dataset, n_cells, n_pairs).
    """
    cell_ids = [f"ACH-{i:04d}" for i in range(n_cells)]
    drugs = ["DrugA", "DrugB"]

    # Synthetic drug-response pairs
    rng = np.random.default_rng(42)
    pair_cell_ids = rng.choice(cell_ids, size=n_pairs)
    pair_drug_names = rng.choice(drugs, size=n_pairs)
    pair_ln_ic50 = rng.standard_normal(n_pairs).astype(np.float32)

    drug_response_df = pd.DataFrame(
        {
            "depmap_id": pair_cell_ids,
            "drug_name": pair_drug_names,
            "ln_ic50": pair_ln_ic50,
        }
    )

    # Synthetic omics — two modalities with 3 and 5 features
    rna_df = pd.DataFrame(
        rng.standard_normal((n_cells, 3)).astype(np.float32),
        index=cell_ids,
        columns=[f"gene_{i}" for i in range(3)],
    )
    mutations_df = pd.DataFrame(
        rng.standard_normal((n_cells, 5)).astype(np.float32),
        index=cell_ids,
        columns=[f"mut_{i}" for i in range(5)],
    )

    parquet_map: dict[str, pd.DataFrame] = {
        "drug_response.parquet": drug_response_df,
        "rna.parquet": rna_df,
        "mutations.parquet": mutations_df,
    }

    def fake_read_parquet(path: Path | str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        name = str(path).split("/")[-1]
        return parquet_map[name]

    with patch("src.data.dataset.pd.read_parquet", side_effect=fake_read_parquet):
        # Also patch path.exists so FileNotFoundError is not raised
        with patch("pathlib.Path.exists", return_value=True):
            ds = MultiOmicsDataset(omics_to_use=["rna", "mutations"])

    return ds, n_cells, n_pairs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToConCatArray:
    def test_shape(self) -> None:
        ds, n_cells, _ = _make_fake_dataset()
        arr = ds.to_concat_array()
        # 3 RNA features + 5 mutation features = 8 total
        assert arr.shape == (n_cells, 8)

    def test_dtype(self) -> None:
        ds, _, _ = _make_fake_dataset()
        arr = ds.to_concat_array()
        assert arr.dtype == np.float32

    def test_contiguous(self) -> None:
        ds, _, _ = _make_fake_dataset()
        arr = ds.to_concat_array()
        assert arr.flags["C_CONTIGUOUS"]

    def test_matches_concat_dim(self) -> None:
        ds, _, _ = _make_fake_dataset()
        arr = ds.to_concat_array()
        assert arr.shape[1] == ds.concat_dim

    def test_returns_new_array_each_call(self) -> None:
        """to_concat_array() should not return a shared mutable reference."""
        ds, _, _ = _make_fake_dataset()
        a1 = ds.to_concat_array()
        a2 = ds.to_concat_array()
        a1[0, 0] += 999.0
        # modifying a1 must not change a2
        assert a2[0, 0] != a1[0, 0]


class TestCellRowsProperty:
    def test_length(self) -> None:
        ds, _, n_pairs = _make_fake_dataset()
        assert len(ds.cell_rows) == n_pairs

    def test_dtype(self) -> None:
        ds, _, _ = _make_fake_dataset()
        assert ds.cell_rows.dtype == np.int64

    def test_values_in_range(self) -> None:
        ds, n_cells, _ = _make_fake_dataset()
        assert ds.cell_rows.min() >= 0
        assert ds.cell_rows.max() < n_cells

    def test_no_private_attribute_access(self) -> None:
        """Confirm cell_rows is a proper property (does not require _cell_rows)."""
        ds, _, _ = _make_fake_dataset()
        # Access via property — must not raise AttributeError
        rows = ds.cell_rows
        assert rows is not None


class TestDrugIndicesProperty:
    def test_length(self) -> None:
        ds, _, n_pairs = _make_fake_dataset()
        assert len(ds.drug_indices) == n_pairs

    def test_dtype(self) -> None:
        ds, _, _ = _make_fake_dataset()
        assert ds.drug_indices.dtype == np.int64

    def test_values_in_range(self) -> None:
        ds, _, _ = _make_fake_dataset()
        assert ds.drug_indices.min() >= 0
        assert ds.drug_indices.max() < ds.n_drugs


class TestTargetsProperty:
    def test_length(self) -> None:
        ds, _, n_pairs = _make_fake_dataset()
        assert len(ds.targets) == n_pairs

    def test_dtype(self) -> None:
        ds, _, _ = _make_fake_dataset()
        assert ds.targets.dtype == np.float32

    def test_consistent_with_get_targets(self) -> None:
        """targets property and get_targets() must return the same values."""
        ds, _, n_pairs = _make_fake_dataset()
        idx = np.arange(n_pairs)
        np.testing.assert_array_equal(ds.targets[idx], ds.get_targets(idx))


class TestNoPrivateAttributesNeeded:
    """Integration check: trainers can work with only public properties."""

    def test_concat_and_index(self) -> None:
        """Simulate what trainer.py does: concat_np[cell_rows[chunk]]."""
        ds, _, n_pairs = _make_fake_dataset()
        concat_np = ds.to_concat_array()
        chunk = np.arange(min(4, n_pairs))
        rows = ds.cell_rows[chunk]
        batch = concat_np[rows]
        assert batch.shape == (len(chunk), ds.concat_dim)

    def test_targets_indexing(self) -> None:
        ds, _, n_pairs = _make_fake_dataset()
        chunk = np.arange(min(4, n_pairs))
        y = ds.targets[chunk]
        assert y.shape == (len(chunk),)
