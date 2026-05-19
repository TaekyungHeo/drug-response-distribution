"""Tests for normalize_binary_fold and normalize_continuous_fold in src/utils/ridge.py."""

from __future__ import annotations

import numpy as np

from src.utils.ridge import normalize_binary_fold, normalize_continuous_fold


class TestNormalizeBinaryFold:
    def _make_binary(self, n_drugs: int = 20, n_features: int = 50, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=(n_drugs, n_features)).astype(np.float32)

    def test_returns_float32(self) -> None:
        fp = self._make_binary()
        train_idx = np.arange(15)
        result, _ = normalize_binary_fold(fp, train_idx)
        assert result.dtype == np.float32

    def test_drops_all_zero_column(self) -> None:
        fp = np.ones((10, 5), dtype=np.float32)
        fp[:, 2] = 0.0  # column 2: no positive bits
        train_idx = np.arange(8)
        _result, n_kept = normalize_binary_fold(fp, train_idx)
        # col_sum for col 2 = 0, which is not > 1 → dropped
        assert n_kept < 5

    def test_drops_all_one_column(self) -> None:
        fp = np.ones((10, 5), dtype=np.float32)
        # col 0: all 1s → col_sum = n_train = 8, NOT < n_train → dropped
        train_idx = np.arange(8)
        _result, n_kept = normalize_binary_fold(fp, train_idx)
        assert n_kept < 5

    def test_keeps_variable_column(self) -> None:
        fp = np.zeros((10, 3), dtype=np.float32)
        fp[0, 0] = 1.0
        fp[1, 0] = 1.0  # col 0: 2 positives out of 8 → kept
        train_idx = np.arange(8)
        _result, n_kept = normalize_binary_fold(fp, train_idx)
        assert n_kept >= 1

    def test_values_unchanged_except_column_filter(self) -> None:
        fp = np.zeros((10, 4), dtype=np.float32)
        # Make col 0 variable
        fp[0, 0] = 1.0
        fp[1, 0] = 1.0
        fp[2, 0] = 1.0
        train_idx = np.arange(8)
        result, _ = normalize_binary_fold(fp, train_idx)
        # Values in kept columns should be the original (0 or 1), not z-scored
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_full_matrix_returned(self) -> None:
        # Returns all n_drugs rows (not just train)
        fp = self._make_binary(n_drugs=20)
        train_idx = np.arange(15)
        result, _ = normalize_binary_fold(fp, train_idx)
        assert result.shape[0] == 20

    def test_no_nan(self) -> None:
        fp = self._make_binary()
        train_idx = np.arange(15)
        result, _ = normalize_binary_fold(fp, train_idx)
        assert not np.any(np.isnan(result))


class TestNormalizeContinuousFold:
    def _make_feat(self, n_drugs: int = 20, n_features: int = 30, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n_drugs, n_features)).astype(np.float32)

    def test_returns_float32(self) -> None:
        feat = self._make_feat()
        train_idx = np.arange(15)
        result = normalize_continuous_fold(feat, train_idx)
        assert result.dtype == np.float32

    def test_train_rows_normalized(self) -> None:
        feat = self._make_feat()
        train_idx = np.arange(15)
        result = normalize_continuous_fold(feat, train_idx)
        # Mean of train rows should be ~0 per feature
        train_means = result[train_idx].mean(axis=0)
        assert np.allclose(train_means, 0.0, atol=1e-5)

    def test_zero_std_column_no_nan(self) -> None:
        feat = self._make_feat()
        feat[:, 3] = 5.0  # constant column
        train_idx = np.arange(15)
        result = normalize_continuous_fold(feat, train_idx)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_full_matrix_returned(self) -> None:
        feat = self._make_feat(n_drugs=20)
        train_idx = np.arange(15)
        result = normalize_continuous_fold(feat, train_idx)
        assert result.shape == feat.shape

    def test_uses_train_statistics_only(self) -> None:
        # Train drugs: values in [0, 1]; test drugs: values in [10, 11]
        # Stats should be fit on train only
        feat = np.zeros((20, 5), dtype=np.float32)
        feat[:15] = np.random.default_rng(0).random((15, 5)).astype(np.float32)
        feat[15:] = 10.0
        train_idx = np.arange(15)
        result = normalize_continuous_fold(feat, train_idx)
        # Test rows (15:) should have large positive values after normalizing by train stats
        assert result[15:].mean() > 5.0
