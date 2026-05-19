"""Tests for src/data/prism.py — preprocess_prism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.prism import preprocess_prism


def _make_df(
    n_drugs: int = 10,
    n_cells: int = 60,
    nan_frac: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal drug-response DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drugs):
        for c in range(n_cells):
            val: float | None = float(rng.normal())
            if rng.random() < nan_frac:
                val = None
            rows.append({"drug_name": f"drug_{d}", "depmap_id": f"cell_{c}", "response": val})
    return pd.DataFrame(rows)


class TestPreprocessPrism:
    def test_returns_tuple_of_three(self) -> None:
        df = _make_df()
        rna_ids = {f"cell_{i}" for i in range(60)}
        result = preprocess_prism(df, rna_ids)
        assert len(result) == 3

    def test_basic_filter(self) -> None:
        df = _make_df(n_drugs=5, n_cells=60)
        rna_ids = {f"cell_{i}" for i in range(60)}
        out_df, n_drugs, n_cells = preprocess_prism(df, rna_ids, min_cells_per_drug=50)
        assert n_drugs == 5
        assert n_cells == 60
        assert isinstance(out_df, pd.DataFrame)

    def test_min_cells_filter(self) -> None:
        df = _make_df(n_drugs=3, n_cells=60)
        # Only half the cells are available
        rna_ids = {f"cell_{i}" for i in range(30)}
        out_df, n_drugs, _ = preprocess_prism(df, rna_ids, min_cells_per_drug=50)
        # 30 cells < 50 threshold → all drugs excluded
        assert n_drugs == 0
        assert len(out_df) == 0

    def test_high_nan_drug_excluded(self) -> None:
        # Build manually: drug_A has >10% NaN, drug_B is clean
        rows = []
        for c in range(60):
            # drug_A: 20% NaN
            val_a: float | None = None if c < 12 else float(c)
            rows.append({"drug_name": "drug_A", "depmap_id": f"cell_{c}", "response": val_a})
            rows.append({"drug_name": "drug_B", "depmap_id": f"cell_{c}", "response": float(c)})
        df = pd.DataFrame(rows)
        rna_ids = {f"cell_{i}" for i in range(60)}
        out_df, n_drugs, _ = preprocess_prism(df, rna_ids, min_cells_per_drug=10)
        assert n_drugs == 1
        assert "drug_B" in out_df["drug_name"].values
        assert "drug_A" not in out_df["drug_name"].values

    def test_cell_intersection(self) -> None:
        df = _make_df(n_drugs=2, n_cells=60)
        # Only first 55 cells available
        rna_ids = {f"cell_{i}" for i in range(55)}
        out_df, _, n_cells = preprocess_prism(df, rna_ids, min_cells_per_drug=10)
        assert all(c in rna_ids for c in out_df["depmap_id"].values)
        assert n_cells <= 55

    def test_no_nan_in_output(self) -> None:
        df = _make_df(nan_frac=0.05)
        rna_ids = {f"cell_{i}" for i in range(60)}
        out_df, _, _ = preprocess_prism(df, rna_ids)
        assert not out_df["response"].isna().any()

    def test_output_columns(self) -> None:
        df = _make_df()
        rna_ids = {f"cell_{i}" for i in range(60)}
        out_df, _, _ = preprocess_prism(df, rna_ids)
        assert "drug_name" in out_df.columns
        assert "depmap_id" in out_df.columns
        assert "response" in out_df.columns
