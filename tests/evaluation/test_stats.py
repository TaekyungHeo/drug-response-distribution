"""Tests for src/evaluation/stats.py — holm_bonferroni and bootstrap_delta_ci."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.stats import bootstrap_delta_ci, holm_bonferroni


class TestHolmBonferroni:
    def test_single_pvalue(self) -> None:
        result = holm_bonferroni({"a": 0.03})
        assert result["a"] == pytest.approx(0.03, abs=1e-9)

    def test_all_significant(self) -> None:
        p_values = {"a": 0.001, "b": 0.002, "c": 0.003}
        result = holm_bonferroni(p_values)
        assert set(result.keys()) == {"a", "b", "c"}
        # All adjusted p should be >= original p
        for k, p in p_values.items():
            assert result[k] >= p

    def test_order_independent(self) -> None:
        p1 = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.10})
        p2 = holm_bonferroni({"c": 0.10, "a": 0.01, "b": 0.04})
        for k in ["a", "b", "c"]:
            assert p1[k] == pytest.approx(p2[k], rel=1e-10)

    def test_capped_at_one(self) -> None:
        # Many tests with high p-values → adjusted p should be capped at 1.0
        p_values = {f"d{i}": 0.5 for i in range(10)}
        result = holm_bonferroni(p_values)
        assert all(v <= 1.0 for v in result.values())

    def test_monotone_increasing(self) -> None:
        # Adjusted p-values should be monotonically non-decreasing when sorted by original p
        p_values = {"a": 0.01, "b": 0.02, "c": 0.05, "d": 0.10}
        result = holm_bonferroni(p_values)
        sorted_keys = sorted(p_values, key=lambda k: p_values[k])
        adjusted_sorted = [result[k] for k in sorted_keys]
        for i in range(len(adjusted_sorted) - 1):
            assert adjusted_sorted[i] <= adjusted_sorted[i + 1] + 1e-10

    def test_known_values(self) -> None:
        # p=[0.01, 0.02, 0.05]; n=3; holm adjusts: 0.01*3=0.03, max(0.02*2=0.04,0.03)=0.04, max(0.05*1=0.05,0.04)=0.05
        result = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.05})
        assert result["a"] == pytest.approx(0.03, abs=1e-9)
        assert result["b"] == pytest.approx(0.04, abs=1e-9)
        assert result["c"] == pytest.approx(0.05, abs=1e-9)


class TestBootstrapDeltaCi:
    def test_returns_two_floats(self) -> None:
        deltas = {f"d{i}": float(i) for i in range(20)}
        lo, hi = bootstrap_delta_ci(deltas, n_bootstrap=200, seed=42)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lo_less_than_hi(self) -> None:
        rng = np.random.default_rng(0)
        deltas = {f"d{i}": float(v) for i, v in enumerate(rng.normal(0.1, 0.5, 50))}
        lo, hi = bootstrap_delta_ci(deltas, n_bootstrap=1000, seed=0)
        assert lo < hi

    def test_covers_true_mean(self) -> None:
        # Symmetric delta distribution centered at 0.2: 95% CI should cover 0.2
        rng = np.random.default_rng(42)
        deltas = {f"d{i}": float(v) for i, v in enumerate(rng.normal(0.2, 0.1, 100))}
        lo, hi = bootstrap_delta_ci(deltas, n_bootstrap=5000, seed=42)
        assert lo < 0.2 < hi

    def test_deterministic_with_seed(self) -> None:
        deltas = {f"d{i}": float(i * 0.01) for i in range(30)}
        lo1, hi1 = bootstrap_delta_ci(deltas, n_bootstrap=500, seed=7)
        lo2, hi2 = bootstrap_delta_ci(deltas, n_bootstrap=500, seed=7)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_single_value(self) -> None:
        # A single drug delta: CI should be [v, v]
        lo, hi = bootstrap_delta_ci({"d": 0.5}, n_bootstrap=100, seed=0)
        assert lo == pytest.approx(0.5, abs=1e-9)
        assert hi == pytest.approx(0.5, abs=1e-9)
