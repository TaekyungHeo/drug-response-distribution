"""Tests for src/utils/solutions.py — MoA, weighted Ridge, K-shot, concordance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.utils.solutions import (
    fit_weighted_ridge,
    group_drugs_by_moa,
    load_moa_annotations,
    pairwise_profile_concordance,
    response_match_predict,
)

_MOA_FILE = Path("external/PASO/Figs/Fig7/GDSC2_Drug_Pathway_Target.csv")
_requires_moa_data = pytest.mark.skipif(
    not _MOA_FILE.exists(), reason=f"data file not present: {_MOA_FILE}"
)

# ── MoA utilities ────────────────────────────────────────────────────────────


class TestMoaUtils:
    @_requires_moa_data
    def test_load_moa_annotations_returns_dict(self):
        moa = load_moa_annotations()
        assert isinstance(moa, dict)
        assert len(moa) > 200
        # Spot-check a well-known drug
        assert "5-Fluorouracil" in moa
        assert moa["5-Fluorouracil"] == "Other"

    def test_group_drugs_by_moa(self):
        moa = {"DrugA": "PI3K", "DrugB": "PI3K", "DrugC": "MAPK", "DrugD": "Other"}
        drugs = ["DrugA", "DrugB", "DrugC", "DrugE"]  # DrugE not in moa
        groups = group_drugs_by_moa(drugs, moa)

        assert "PI3K" in groups
        assert set(groups["PI3K"]) == {"DrugA", "DrugB"}
        assert groups["MAPK"] == ["DrugC"]
        # DrugD not in drug_names, DrugE not in moa -> neither appears
        assert "Other" not in groups
        assert "DrugE" not in [d for ds in groups.values() for d in ds]


# ── Weighted Ridge ───────────────────────────────────────────────────────────


class TestWeightedRidge:
    def test_unweighted_matches_sklearn(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ rng.randn(5) + 0.1 * rng.randn(100)

        ours = fit_weighted_ridge(X, y, sample_weights=None, alpha=1.0)
        ref = Ridge(alpha=1.0, fit_intercept=True).fit(X, y)

        np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-10)
        np.testing.assert_allclose(ours.intercept_, ref.intercept_, atol=1e-10)

    def test_weighted_changes_result(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 3)
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.randn(100)

        uniform = fit_weighted_ridge(X, y, sample_weights=None, alpha=1.0)

        # Heavily upweight first half
        w = np.ones(100)
        w[:50] = 10.0
        weighted = fit_weighted_ridge(X, y, sample_weights=w, alpha=1.0)

        # Coefficients should differ
        assert not np.allclose(uniform.coef_, weighted.coef_, atol=1e-4)


# ── Response matching ────────────────────────────────────────────────────────


class TestResponseMatching:
    @pytest.fixture()
    def setup_data(self):
        rng = np.random.RandomState(123)
        n_train, n_cells = 50, 80
        train_mat = rng.randn(n_train, n_cells).astype(np.float32)
        cell_mean = train_mat.mean(axis=0).astype(np.float64)
        return train_mat, cell_mean

    def test_k0_returns_cell_mean(self, setup_data):
        train_mat, cell_mean = setup_data
        pred = response_match_predict(
            train_mat,
            test_observed=np.array([]),
            anchor_cell_idx=np.array([], dtype=int),
            cell_mean=cell_mean,
        )
        np.testing.assert_allclose(pred, cell_mean)

    def test_k50_differs_from_cell_mean(self, setup_data):
        train_mat, cell_mean = setup_data
        rng = np.random.RandomState(99)

        # Generate a test drug that is correlated with training drug 0
        anchor_idx = np.arange(50)
        test_obs = train_mat[0, anchor_idx] + 0.1 * rng.randn(50)

        pred = response_match_predict(
            train_mat,
            test_observed=test_obs,
            anchor_cell_idx=anchor_idx,
            cell_mean=cell_mean,
            blend_weight=0.8,
            n_neighbors=5,
        )
        # With strong signal, prediction should differ from cell_mean
        assert not np.allclose(pred, cell_mean, atol=0.1)

    def test_permuted_control_near_cell_mean(self, setup_data):
        """With shuffled observations, neighbor signal is noise -> close to cell_mean."""
        train_mat, cell_mean = setup_data
        rng = np.random.RandomState(77)

        anchor_idx = np.arange(50)
        # Random noise uncorrelated with any training drug
        test_obs = rng.randn(50)

        pred = response_match_predict(
            train_mat,
            test_observed=test_obs,
            anchor_cell_idx=anchor_idx,
            cell_mean=cell_mean,
            blend_weight=0.5,
            n_neighbors=5,
        )
        # With blend_weight=0.5 and noisy neighbors, deviation from cell_mean
        # should be much smaller than the signal we see in the K=50 test
        diff = np.abs(pred - cell_mean).mean()
        # The deviation should be moderate — certainly under 0.5
        assert diff < 0.5


# ── Profile concordance ──────────────────────────────────────────────────────


class TestProfileConcordance:
    def test_identical_profiles_give_r1(self):
        # Two drugs with identical profiles -> r = 1.0
        mat = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0] * 5,  # 25 cells
                [1.0, 2.0, 3.0, 4.0, 5.0] * 5,
                [5.0, 4.0, 3.0, 2.0, 1.0] * 5,  # different profile
            ]
        )
        names = ["D1", "D2", "D3"]
        groups = {"GroupA": ["D1", "D2"], "GroupB": ["D1", "D3"]}

        result = pairwise_profile_concordance(mat, names, groups, min_shared_cells=20)

        assert "GroupA" in result
        np.testing.assert_allclose(result["GroupA"]["mean_r"], 1.0, atol=1e-10)
        assert result["GroupA"]["n_pairs"] == 1
        assert result["GroupA"]["n_drugs"] == 2

        # D1 and D3 are anti-correlated
        assert "GroupB" in result
        np.testing.assert_allclose(result["GroupB"]["mean_r"], -1.0, atol=1e-10)

    def test_small_group_excluded(self):
        mat = np.random.RandomState(42).randn(3, 30)
        names = ["D1", "D2", "D3"]
        # Only 1 drug in group -> excluded
        groups = {"Solo": ["D1"], "Pair": ["D2", "D3"]}

        result = pairwise_profile_concordance(mat, names, groups, min_shared_cells=20)

        assert "Solo" not in result
        assert "Pair" in result
