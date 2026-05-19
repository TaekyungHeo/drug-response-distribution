"""Tests for src/data/omics_utils.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.omics_utils import build_pair_features, z_score_normalize

# ── build_pair_features ────────────────────────────────────────────────


class TestBuildPairFeatures:
    def _make_data(self) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        cell_order = ["ACH-001", "ACH-002", "ACH-003"]
        cell_mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        pairs = pd.DataFrame({
            "depmap_id": ["ACH-001", "ACH-003", "ACH-002", "ACH-001"],
            "drug_name": ["DrugA", "DrugB", "DrugA", "DrugB"],
        })
        return pairs, cell_mat, cell_order

    def test_shape(self) -> None:
        pairs, cell_mat, cell_order = self._make_data()
        out = build_pair_features(pairs, cell_mat, cell_order)
        assert out.shape == (4, 2)

    def test_correct_rows_extracted(self) -> None:
        pairs, cell_mat, cell_order = self._make_data()
        out = build_pair_features(pairs, cell_mat, cell_order)
        # First pair: ACH-001 → row 0 → [1.0, 2.0]
        np.testing.assert_array_equal(out[0], [1.0, 2.0])
        # Second pair: ACH-003 → row 2 → [5.0, 6.0]
        np.testing.assert_array_equal(out[1], [5.0, 6.0])
        # Third pair: ACH-002 → row 1 → [3.0, 4.0]
        np.testing.assert_array_equal(out[2], [3.0, 4.0])

    def test_dtype_preserved(self) -> None:
        pairs, cell_mat, cell_order = self._make_data()
        out = build_pair_features(pairs, cell_mat, cell_order)
        assert out.dtype == np.float32

    def test_single_pair(self) -> None:
        cell_order = ["ACH-001"]
        cell_mat = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
        pairs = pd.DataFrame({"depmap_id": ["ACH-001"], "drug_name": ["DrugX"]})
        out = build_pair_features(pairs, cell_mat, cell_order)
        assert out.shape == (1, 3)
        np.testing.assert_array_equal(out[0], [7.0, 8.0, 9.0])

    def test_repeated_cell(self) -> None:
        """Same cell appearing in multiple pairs gets the same feature row."""
        cell_order = ["ACH-001", "ACH-002"]
        cell_mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pairs = pd.DataFrame({
            "depmap_id": ["ACH-001", "ACH-001", "ACH-001"],
            "drug_name": ["D1", "D2", "D3"],
        })
        out = build_pair_features(pairs, cell_mat, cell_order)
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(out[0], out[1])
        np.testing.assert_array_equal(out[0], out[2])


# ── z_score_normalize ─────────────────────────────────────────────────


class TestZScoreNormalize:
    def _make_arrays(
        self, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X_train = rng.standard_normal((100, 10)).astype(np.float32) * 3 + 5
        X_val = rng.standard_normal((20, 10)).astype(np.float32) * 3 + 5
        X_test = rng.standard_normal((30, 10)).astype(np.float32) * 3 + 5
        return X_train, X_val, X_test

    def test_output_shapes_unchanged(self) -> None:
        X_tr, X_va, X_te = self._make_arrays()
        tr, va, te = z_score_normalize(X_tr, X_va, X_te)
        assert tr.shape == X_tr.shape
        assert va.shape == X_va.shape
        assert te.shape == X_te.shape

    def test_train_mean_near_zero(self) -> None:
        X_tr, X_va, X_te = self._make_arrays()
        tr, _, _ = z_score_normalize(X_tr, X_va, X_te)
        np.testing.assert_allclose(tr.mean(axis=0), 0.0, atol=1e-5)

    def test_train_std_near_one(self) -> None:
        X_tr, X_va, X_te = self._make_arrays()
        tr, _, _ = z_score_normalize(X_tr, X_va, X_te)
        np.testing.assert_allclose(tr.std(axis=0), 1.0, atol=1e-4)

    def test_uses_train_stats_for_val_test(self) -> None:
        """Val/test normalised with train mean/std, not their own."""
        X_tr = np.array([[0.0, 10.0], [2.0, 10.0]], dtype=np.float32)
        X_va = np.array([[10.0, 10.0]], dtype=np.float32)
        X_te = np.array([[10.0, 10.0]], dtype=np.float32)
        _tr, va, _te = z_score_normalize(X_tr, X_va, X_te)
        # Train mean col0 = 1.0, std = 1.0 → X_val[0,0] = (10-1)/1 = 9
        assert abs(float(va[0, 0]) - 9.0) < 1e-4

    def test_zero_std_column_clamped(self) -> None:
        """Constant column should not produce NaN or Inf."""
        X_tr = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        X_va = np.array([[5.0, 5.0]], dtype=np.float32)
        X_te = np.array([[5.0, 5.0]], dtype=np.float32)
        tr, va, _te = z_score_normalize(X_tr, X_va, X_te)
        assert not np.any(np.isnan(tr))
        assert not np.any(np.isinf(tr))
        assert not np.any(np.isnan(va))

    def test_output_dtype_float32(self) -> None:
        X_tr, X_va, X_te = self._make_arrays()
        tr, va, te = z_score_normalize(X_tr, X_va, X_te)
        assert tr.dtype == np.float32
        assert va.dtype == np.float32
        assert te.dtype == np.float32

    def test_no_nan_in_output(self) -> None:
        X_tr, X_va, X_te = self._make_arrays()
        tr, va, te = z_score_normalize(X_tr, X_va, X_te)
        assert not np.any(np.isnan(tr))
        assert not np.any(np.isnan(va))
        assert not np.any(np.isnan(te))
