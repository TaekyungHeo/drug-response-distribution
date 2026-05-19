"""Unit tests for src/utils/response_matching.py — pure in-memory, no disk I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.response_matching import (
    build_response_matrix,
    response_match,
    select_cells_diverse,
    select_cells_maxvar,
    select_cells_midresp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUGS = ["drugA", "drugB", "drugC", "drugD", "drugE"]
CELLS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]


@pytest.fixture()
def response_df() -> pd.DataFrame:
    """5 drugs x 8 cells long-format DataFrame with known values."""
    rng = np.random.default_rng(42)
    rows = []
    for d in DRUGS:
        for c in CELLS:
            rows.append({"drug_name": d, "depmap_id": c, "ln_ic50": float(rng.normal(0, 1))})
    return pd.DataFrame(rows)


@pytest.fixture()
def response_mat(response_df: pd.DataFrame) -> np.ndarray:
    return build_response_matrix(response_df, DRUGS, CELLS)


# ---------------------------------------------------------------------------
# build_response_matrix
# ---------------------------------------------------------------------------


class TestBuildResponseMatrix:
    def test_shape(self, response_df: pd.DataFrame) -> None:
        mat = build_response_matrix(response_df, DRUGS, CELLS)
        assert mat.shape == (len(DRUGS), len(CELLS))

    def test_dtype(self, response_df: pd.DataFrame) -> None:
        mat = build_response_matrix(response_df, DRUGS, CELLS)
        assert mat.dtype == np.float32

    def test_nan_for_missing(self) -> None:
        """Drugs/cells not in the DataFrame should be NaN."""
        df = pd.DataFrame([{"drug_name": "drugA", "depmap_id": "c0", "ln_ic50": 1.0}])
        mat = build_response_matrix(df, ["drugA", "drugB"], ["c0", "c1"])
        assert mat[0, 0] == pytest.approx(1.0)
        assert np.isnan(mat[0, 1])
        assert np.isnan(mat[1, 0])
        assert np.isnan(mat[1, 1])


# ---------------------------------------------------------------------------
# response_match
# ---------------------------------------------------------------------------


class TestResponseMatch:
    def test_k0_returns_column_means(self, response_mat: np.ndarray) -> None:
        pred_cells = np.array([0, 1, 2])
        preds = response_match(
            obs_cells=np.array([], dtype=int),
            obs_vals=np.array([], dtype=np.float32),
            train_mat=response_mat,
            pred_cells=pred_cells,
            K=0,
        )
        expected = np.nanmean(response_mat[:, pred_cells], axis=0)
        np.testing.assert_allclose(preds, expected, rtol=1e-5)

    def test_k1_nearest_potency(self) -> None:
        """K=1 should match by nearest IC50 in the observed cell column."""
        train_mat = np.array(
            [[1.0, 2.0], [5.0, 6.0], [1.1, 2.1], [10.0, 11.0]],
            dtype=np.float32,
        )
        # Observe cell 0 with value 1.05 -> closest rows are 0 and 2
        preds = response_match(
            obs_cells=np.array([0]),
            obs_vals=np.array([1.05], dtype=np.float32),
            train_mat=train_mat,
            pred_cells=np.array([1]),
            K=1,
            top_n=2,
        )
        # Weighted equally -> mean of col1 rows 0,2 = (2.0+2.1)/2
        assert preds[0] == pytest.approx(2.05, abs=0.01)

    def test_k2_correlation(self) -> None:
        """K>=2 should rank training drugs by Pearson correlation."""
        rng = np.random.default_rng(7)
        n_train, n_cells = 20, 10
        train_mat = rng.normal(size=(n_train, n_cells)).astype(np.float32)
        obs_cells = np.array([0, 1, 2, 3])
        # Use a known drug profile as observation
        obs_vals = train_mat[5, obs_cells] + rng.normal(0, 0.01, size=4).astype(np.float32)
        pred_cells = np.array([4, 5, 6])
        preds = response_match(obs_cells, obs_vals, train_mat, pred_cells, K=4, top_n=3)
        assert preds.shape == (3,)
        assert np.all(np.isfinite(preds))

    def test_all_nan_train_mat(self) -> None:
        """All-NaN training matrix should not crash."""
        train_mat = np.full((5, 8), np.nan, dtype=np.float32)
        preds = response_match(
            obs_cells=np.array([0, 1]),
            obs_vals=np.array([1.0, 2.0], dtype=np.float32),
            train_mat=train_mat,
            pred_cells=np.array([2, 3]),
            K=2,
        )
        assert preds.shape == (2,)


# ---------------------------------------------------------------------------
# Golden tests — hand-computed expected values
# ---------------------------------------------------------------------------


class TestResponseMatchGolden:
    """Verify exact outputs against hand-computed values on tiny inputs."""

    @pytest.fixture()
    def tiny_train(self) -> np.ndarray:
        """3 training drugs × 4 cells, no NaN."""
        return np.array(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [1.2, 2.1, 3.3, 3.9]],
            dtype=np.float32,
        )

    def test_k0_golden(self, tiny_train: np.ndarray) -> None:
        """K=0 must return column means of training matrix."""
        pred_cells = np.array([0, 1, 2, 3])
        preds = response_match(
            np.array([], dtype=int), np.array([], dtype=np.float32),
            tiny_train, pred_cells, K=0,
        )
        expected = tiny_train.mean(axis=0)
        np.testing.assert_allclose(preds, expected, atol=1e-5)

    def test_k1_golden_nearest_ic50(self, tiny_train: np.ndarray) -> None:
        """K=1: observe cell 0 with value 1.1 → nearest are drug0 (1.0) and drug2 (1.2).
        With top_n=2, equal weights → predicted cell 1 = mean(2.0, 2.1) = 2.05."""
        preds = response_match(
            obs_cells=np.array([0]),
            obs_vals=np.array([1.1], dtype=np.float32),
            train_mat=tiny_train,
            pred_cells=np.array([1, 2]),
            K=1, top_n=2,
        )
        assert preds[0] == pytest.approx(2.05, abs=0.01)
        assert preds[1] == pytest.approx(3.15, abs=0.01)

    def test_k1_does_use_observation(self, tiny_train: np.ndarray) -> None:
        """K=1 with different observation values should give different predictions."""
        pred_cells = np.array([2, 3])
        preds_low = response_match(
            np.array([0]), np.array([1.0], dtype=np.float32),
            tiny_train, pred_cells, K=1, top_n=2,
        )
        preds_high = response_match(
            np.array([0]), np.array([5.0], dtype=np.float32),
            tiny_train, pred_cells, K=1, top_n=2,
        )
        # Observation of 1.0 matches drug0/drug2; observation of 5.0 matches drug1
        assert not np.allclose(preds_low, preds_high, atol=0.1)

    def test_k1_not_equal_to_k0(self, tiny_train: np.ndarray) -> None:
        """K=1 must differ from K=0 (proves observation is used, not cell-mean)."""
        pred_cells = np.array([1, 2, 3])
        k0_preds = response_match(
            np.array([], dtype=int), np.array([], dtype=np.float32),
            tiny_train, pred_cells, K=0,
        )
        k1_preds = response_match(
            np.array([0]), np.array([1.0], dtype=np.float32),
            tiny_train, pred_cells, K=1, top_n=2,
        )
        assert not np.allclose(k0_preds, k1_preds, atol=0.01)

    def test_k3_correlation_ranking(self) -> None:
        """K=3: drug0's profile is nearly identical to obs → should rank highest."""
        train_mat = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0],
             [5.0, 4.0, 3.0, 2.0, 1.0],
             [1.0, 2.0, 3.0, 4.0, 5.0]],  # same as drug0
            dtype=np.float32,
        )
        obs_cells = np.array([0, 1, 2])
        obs_vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pred_cells = np.array([3, 4])
        preds = response_match(obs_cells, obs_vals, train_mat, pred_cells, K=3, top_n=2)
        # drug0 and drug2 (identical profiles) should dominate
        np.testing.assert_allclose(preds, [4.0, 5.0], atol=0.01)


# ---------------------------------------------------------------------------
# select_cells_*
# ---------------------------------------------------------------------------


class TestCellSelection:
    def test_maxvar_picks_highest_variance(self) -> None:
        mat = np.array([[1, 2, 100], [1, 2, -100], [1, 2, 0]], dtype=np.float32)
        valid = np.arange(3)
        picked = select_cells_maxvar(1, valid, mat)
        assert picked[0] == 2  # col 2 has highest variance

    def test_midresp_picks_median_cells(self) -> None:
        mat = np.array([[0, 5, 10], [0, 5, 10], [0, 5, 10]], dtype=np.float32)
        valid = np.arange(3)
        picked = select_cells_midresp(1, valid, mat)
        assert picked[0] == 1  # col 1 median=5 is closest to grand median=5

    def test_diverse_k1_equals_maxvar(self) -> None:
        rng = np.random.default_rng(0)
        mat = rng.normal(size=(10, 6)).astype(np.float32)
        valid = np.arange(6)
        div = select_cells_diverse(1, valid, mat)
        mv = select_cells_maxvar(1, valid, mat)
        np.testing.assert_array_equal(div, mv)

    def test_diverse_k_gt1_distinct(self) -> None:
        rng = np.random.default_rng(1)
        mat = rng.normal(size=(10, 8)).astype(np.float32)
        valid = np.arange(8)
        picked = select_cells_diverse(3, valid, mat)
        assert len(picked) == 3
        assert len(set(picked)) == 3  # all distinct
