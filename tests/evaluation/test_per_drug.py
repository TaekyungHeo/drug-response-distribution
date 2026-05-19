"""Tests for per_drug evaluation utilities (src/evaluation/per_drug.py)
and ridge regression helpers (src/utils/ridge.py).

These functions are used in 14–15 experiment scripts. Testing the canonical
versions here prevents silent divergence when experiments are updated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.per_drug import (
    mean_per_cell_r,
    mean_per_drug_r,
    per_cell_r,
    per_drug_r,
    per_moa_r,
)
from src.utils.ridge import compress_cell, compress_multi_omics, safe_fit_scaler

# ── per_drug_r ─────────────────────────────────────────────────────────


class TestPerDrugR:
    def test_basic_correlation(self) -> None:
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        drugs = np.array(["A", "A", "A", "A", "A"])
        rs = per_drug_r(preds, targets, drugs)
        assert "A" in rs
        assert rs["A"] > 0.99

    def test_filters_below_min_cells(self) -> None:
        # Drug B has only 2 samples, should be excluded
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        drugs = np.array(["A", "A", "A", "A", "A", "B", "B"])
        rs = per_drug_r(preds, targets, drugs, min_cells=5)
        assert "A" in rs
        assert "B" not in rs

    def test_constant_prediction_returns_nan_excluded(self) -> None:
        # If predictions are constant, std < 1e-8 → excluded
        preds = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        drugs = np.array(["A"] * 5)
        rs = per_drug_r(preds, targets, drugs)
        assert "A" not in rs

    def test_two_drugs_independent(self) -> None:
        preds = np.array([1, 2, 3, 4, 5, -1, -2, -3, -4, -5], dtype=float)
        targets = np.array([1.1, 1.9, 3.1, 3.9, 5.1, -1, -2, -3, -4, -5], dtype=float)
        drugs = np.array(["A"] * 5 + ["B"] * 5)
        rs = per_drug_r(preds, targets, drugs)
        assert set(rs.keys()) == {"A", "B"}
        assert rs["A"] > 0.99
        assert rs["B"] > 0.99

    def test_spearman_metric(self) -> None:
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        drugs = np.array(["A"] * 5)
        rs = per_drug_r(preds, targets, drugs, metric="spearman")
        assert "A" in rs
        assert abs(rs["A"] - 1.0) < 1e-6

    def test_empty_result_for_all_filtered(self) -> None:
        preds = np.array([1.0, 2.0])
        targets = np.array([1.0, 2.0])
        drugs = np.array(["A", "A"])
        rs = per_drug_r(preds, targets, drugs, min_cells=5)
        assert rs == {}


class TestPerDrugRGolden:
    """Hand-computed golden values."""

    def test_known_correlation(self) -> None:
        """Drug A: preds=[1,2,3,4,5], targets=[2,4,6,8,10] → r=1.0 exactly."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        drugs = np.array(["A"] * 5)
        rs = per_drug_r(preds, targets, drugs)
        assert abs(rs["A"] - 1.0) < 1e-10

    def test_negative_correlation(self) -> None:
        """Perfect negative correlation → r=-1.0."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        drugs = np.array(["A"] * 5)
        rs = per_drug_r(preds, targets, drugs)
        assert abs(rs["A"] - (-1.0)) < 1e-10

    def test_multi_drug_independence(self) -> None:
        """Two drugs computed independently; mean = (1.0 + (-1.0)) / 2 = 0."""
        preds = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1], dtype=float)
        targets = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
        drugs = np.array(["A"] * 5 + ["B"] * 5)
        rs = per_drug_r(preds, targets, drugs)
        assert abs(rs["A"] - 1.0) < 1e-10
        assert abs(rs["B"] - (-1.0)) < 1e-10
        m = mean_per_drug_r(preds, targets, drugs)
        assert abs(m) < 1e-10


class TestMeanPerDrugR:
    def test_mean_over_multiple_drugs(self) -> None:
        preds = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
        targets = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1], dtype=float)
        drugs = np.array(["A"] * 5 + ["B"] * 5)
        m = mean_per_drug_r(preds, targets, drugs)
        # Drug A: r=1.0, Drug B: r=-1.0 → mean = 0.0
        assert abs(m) < 1e-6

    def test_returns_nan_when_empty(self) -> None:
        preds = np.array([1.0, 2.0])
        targets = np.array([1.0, 2.0])
        drugs = np.array(["A", "A"])
        m = mean_per_drug_r(preds, targets, drugs, min_cells=5)
        assert np.isnan(m)


class TestPerMoaR:
    def test_groups_by_moa(self) -> None:
        preds = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
        targets = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
        drugs = np.array(["DrugA"] * 5 + ["DrugB"] * 5)
        drug_moa = {"DrugA": "MoA1", "DrugB": "MoA2"}
        rs = per_moa_r(preds, targets, drugs, drug_moa)
        assert "MoA1" in rs
        assert "MoA2" in rs
        assert abs(rs["MoA1"] - 1.0) < 1e-6

    def test_focus_moa_filter(self) -> None:
        preds = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
        targets = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
        drugs = np.array(["DrugA"] * 5 + ["DrugB"] * 5)
        drug_moa = {"DrugA": "MoA1", "DrugB": "MoA2"}
        rs = per_moa_r(preds, targets, drugs, drug_moa, focus_moa=["MoA1"])
        assert "MoA1" in rs
        assert "MoA2" not in rs


# ── safe_fit_scaler ────────────────────────────────────────────────────


class TestSafeFitScaler:
    def test_normal_column_scaled(self) -> None:
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        sc = safe_fit_scaler(X)
        Xt = sc.transform(X)
        # column 0 should have mean 0, std 1
        assert abs(Xt[:, 0].mean()) < 1e-6
        assert abs(Xt[:, 0].std() - 1.0) < 0.1

    def test_zero_variance_column_not_nan(self) -> None:
        # Column 1 is constant
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        sc = safe_fit_scaler(X)
        Xt = sc.transform(X)
        assert not np.any(np.isnan(Xt))
        assert not np.any(np.isinf(Xt))

    def test_scale_clamped_at_one(self) -> None:
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        sc = safe_fit_scaler(X)
        assert sc.scale_[1] == 1.0  # clamped from 0


# ── compress_cell ──────────────────────────────────────────────────────


class TestCompressCell:
    def _make_data(
        self, n_cells: int = 50, n_rna: int = 100, n_mut: int = 20, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        rna = rng.random((n_cells, n_rna), dtype=np.float32)
        mut = rng.random((n_cells, n_mut), dtype=np.float32)
        return rna, mut

    def test_output_shapes(self) -> None:
        rna, mut = self._make_data()
        train_rows = np.arange(40)
        rna_r, mut_r = compress_cell(rna, mut, train_rows, rna_dim=10, mut_dim=5)
        assert rna_r.shape == (50, 10)
        assert mut_r.shape == (50, 5)

    def test_output_dtype_float32(self) -> None:
        rna, mut = self._make_data()
        train_rows = np.arange(40)
        rna_r, mut_r = compress_cell(rna, mut, train_rows)
        assert rna_r.dtype == np.float32
        assert mut_r.dtype == np.float32

    def test_respects_rna_dim_limit(self) -> None:
        # Only 10 training cells → can't have more than 9 components
        rna, mut = self._make_data(n_cells=20)
        train_rows = np.arange(10)
        rna_r, _mut_r = compress_cell(rna, mut, train_rows, rna_dim=550, mut_dim=200)
        assert rna_r.shape[1] <= 9  # min(550, 10-1, 100)

    def test_no_nan_in_output(self) -> None:
        rna, mut = self._make_data()
        train_rows = np.arange(40)
        rna_r, mut_r = compress_cell(rna, mut, train_rows)
        assert not np.any(np.isnan(rna_r))
        assert not np.any(np.isnan(mut_r))


# ── per_cell_r ─────────────────────────────────────────────────────────


class TestPerCellR:
    def test_basic_correlation(self) -> None:
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        cells = np.array(["C1"] * 5)
        rs = per_cell_r(preds, targets, cells)
        assert "C1" in rs
        assert rs["C1"] > 0.99

    def test_filters_below_min_drugs(self) -> None:
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        cells = np.array(["C1", "C1", "C1", "C1", "C1", "C2", "C2"])
        rs = per_cell_r(preds, targets, cells, min_drugs=5)
        assert "C1" in rs
        assert "C2" not in rs

    def test_constant_prediction_excluded(self) -> None:
        preds = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cells = np.array(["C1"] * 5)
        rs = per_cell_r(preds, targets, cells)
        assert "C1" not in rs

    def test_two_cells_independent(self) -> None:
        preds = np.array([1, 2, 3, 4, 5, -1, -2, -3, -4, -5], dtype=float)
        targets = np.array([1.1, 1.9, 3.1, 3.9, 5.1, -1, -2, -3, -4, -5], dtype=float)
        cells = np.array(["C1"] * 5 + ["C2"] * 5)
        rs = per_cell_r(preds, targets, cells)
        assert set(rs.keys()) == {"C1", "C2"}
        assert rs["C1"] > 0.99
        assert rs["C2"] > 0.99

    def test_perfect_negative_correlation(self) -> None:
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        cells = np.array(["C1"] * 5)
        rs = per_cell_r(preds, targets, cells)
        assert abs(rs["C1"] - (-1.0)) < 1e-10

    def test_spearman_metric(self) -> None:
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cells = np.array(["C1"] * 5)
        rs = per_cell_r(preds, targets, cells, metric="spearman")
        assert abs(rs["C1"] - 1.0) < 1e-6

    def test_empty_result_for_all_filtered(self) -> None:
        preds = np.array([1.0, 2.0])
        targets = np.array([1.0, 2.0])
        cells = np.array(["C1", "C1"])
        rs = per_cell_r(preds, targets, cells, min_drugs=5)
        assert rs == {}


class TestMeanPerCellR:
    def test_mean_over_multiple_cells(self) -> None:
        # C1: r=1.0, C2: r=-1.0 → mean = 0.0
        preds = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1], dtype=float)
        targets = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
        cells = np.array(["C1"] * 5 + ["C2"] * 5)
        m = mean_per_cell_r(preds, targets, cells)
        assert abs(m) < 1e-6

    def test_returns_nan_when_empty(self) -> None:
        preds = np.array([1.0, 2.0])
        targets = np.array([1.0, 2.0])
        cells = np.array(["C1", "C1"])
        m = mean_per_cell_r(preds, targets, cells, min_drugs=5)
        assert np.isnan(m)

    def test_single_cell_equals_per_cell(self) -> None:
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        cells = np.array(["C1"] * 5)
        m = mean_per_cell_r(preds, targets, cells)
        assert abs(m - 1.0) < 1e-10


# ── compress_multi_omics ───────────────────────────────────────────────


class TestCompressMultiOmics:
    def _make_omics(
        self, n_cells: int = 50, seed: int = 0
    ) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        cells = [f"C{i:03d}" for i in range(n_cells)]
        return {
            "rna": pd.DataFrame(rng.random((n_cells, 100), dtype=np.float32), index=cells),
            "mutations": pd.DataFrame(rng.random((n_cells, 30), dtype=np.float32), index=cells),
            "cnv": pd.DataFrame(rng.random((n_cells, 60), dtype=np.float32), index=cells),
            "rppa": pd.DataFrame(rng.random((n_cells, 20), dtype=np.float32), index=cells),
        }

    def test_output_shape_rna_mut(self) -> None:
        omics = self._make_omics()
        all_cells = list(omics["rna"].index)
        train_cells = all_cells[:40]
        feat, cell_to_row = compress_multi_omics(
            omics, ["rna", "mutations"], all_cells, train_cells,
            pca_dims={"rna": 10, "mutations": 5},
        )
        assert feat.shape == (50, 15)
        assert len(cell_to_row) == 50

    def test_raw_modality_passthrough(self) -> None:
        omics = self._make_omics()
        all_cells = list(omics["rna"].index)
        train_cells = all_cells[:40]
        feat, _ = compress_multi_omics(
            omics, ["rna", "rppa"], all_cells, train_cells,
            pca_dims={"rna": 5},
        )
        # rppa has 20 features, no PCA → dim = 5 + 20
        assert feat.shape[1] == 25

    def test_cell_to_row_mapping(self) -> None:
        omics = self._make_omics(n_cells=10)
        all_cells = list(omics["rna"].index)
        _, cell_to_row = compress_multi_omics(
            omics, ["rna"], all_cells, all_cells[:8],
            pca_dims={"rna": 3},
        )
        assert cell_to_row[all_cells[0]] == 0
        assert cell_to_row[all_cells[-1]] == len(all_cells) - 1

    def test_pca_fit_on_train_only(self) -> None:
        omics = self._make_omics()
        all_cells = list(omics["rna"].index)
        train_cells = all_cells[:40]
        # Should not raise even though test cells weren't used for PCA fit
        feat, _ = compress_multi_omics(
            omics, ["rna", "mutations", "cnv"], all_cells, train_cells,
            pca_dims={"rna": 10, "mutations": 5, "cnv": 8},
        )
        assert feat.shape == (50, 23)
        assert not np.any(np.isnan(feat))

    def test_output_dtype_float32(self) -> None:
        omics = self._make_omics()
        all_cells = list(omics["rna"].index)
        feat, _ = compress_multi_omics(
            omics, ["rna", "mutations"], all_cells, all_cells[:40],
            pca_dims={"rna": 5, "mutations": 3},
        )
        assert feat.dtype == np.float32

    def test_n_components_capped_by_train_cells(self) -> None:
        omics = self._make_omics(n_cells=15)
        all_cells = list(omics["rna"].index)
        train_cells = all_cells[:8]
        feat, _ = compress_multi_omics(
            omics, ["rna"], all_cells, train_cells,
            pca_dims={"rna": 100},  # more than n_train - 1
        )
        assert feat.shape[1] <= 7  # capped at len(unique train rows) - 1
