"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import evaluate, evaluate_full, pearson_r, rmse


def test_pearson_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pearson_r(y, y) == pytest.approx(1.0)


def test_pearson_anti() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pearson_r(y, -y) == pytest.approx(-1.0)


def test_rmse_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_evaluate_returns_all_keys() -> None:
    y = np.array([1.0, 2.0, 3.0])
    result = evaluate(y, y)
    assert set(result.keys()) == {"pearson_r", "spearman_r", "rmse", "n"}
    assert result["n"] == 3


def test_pearson_too_few_samples() -> None:
    assert np.isnan(pearson_r(np.array([1.0]), np.array([1.0])))


# ── evaluate_full ──────────────────────────────────────────────────────


class TestEvaluateFull:
    def _make_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        n = 20
        y_true = rng.standard_normal(n).astype(np.float32)
        y_pred = y_true + rng.standard_normal(n).astype(np.float32) * 0.1
        drugs = np.array(["DrugA"] * 10 + ["DrugB"] * 10)
        cells = np.array([f"C{i}" for i in range(10)] * 2)
        return y_true, y_pred, drugs, cells

    def test_returns_required_keys(self) -> None:
        y_true, y_pred, drugs, cells = self._make_data()
        result = evaluate_full(y_true, y_pred, drugs, cells)
        assert set(result.keys()) == {"global_r", "per_drug_r", "per_cell_r", "n"}

    def test_n_equals_input_length(self) -> None:
        y_true, y_pred, drugs, cells = self._make_data()
        result = evaluate_full(y_true, y_pred, drugs, cells)
        assert result["n"] == len(y_true)

    def test_global_r_near_one_for_near_perfect_preds(self) -> None:
        y_true, y_pred, drugs, cells = self._make_data()
        result = evaluate_full(y_true, y_pred, drugs, cells)
        assert float(result["global_r"]) > 0.9  # type: ignore[arg-type]

    def test_perfect_predictions_per_drug_one(self) -> None:
        # 2 drugs × 5 cells — each drug has 5 samples → per_drug_r = 1.0
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        drugs = np.array(["A"] * 5 + ["B"] * 5)
        cells = np.array(["C1", "C2", "C3", "C4", "C5"] * 2)
        result = evaluate_full(y, y, drugs, cells)
        assert abs(float(result["global_r"]) - 1.0) < 1e-6  # type: ignore[arg-type]
        assert abs(float(result["per_drug_r"]) - 1.0) < 1e-6  # type: ignore[arg-type]

    def test_global_r_uses_pearson(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 2, dtype=float)
        y_pred = np.array([2.0, 4.0, 6.0, 8.0, 10.0] * 2, dtype=float)
        drugs = np.array(["A"] * 5 + ["B"] * 5)
        cells = np.array(["C1", "C2", "C3", "C4", "C5"] * 2)
        result = evaluate_full(y_true, y_pred, drugs, cells)
        assert abs(float(result["global_r"]) - 1.0) < 1e-6  # type: ignore[arg-type]
