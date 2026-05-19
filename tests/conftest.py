"""Shared fixtures for the multi-onco test suite.

Modern pytest best practice: shared fixtures live in conftest.py,
discovered automatically by pytest. No need to import them in test files.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ── Model fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def feature_dims() -> dict[str, int]:
    """Small feature dimensions for fast model tests."""
    return {"rna": 16, "mutations": 8}


@pytest.fixture()
def modality_order() -> list[str]:
    return ["rna", "mutations"]


@pytest.fixture()
def batch_size() -> int:
    return 4


@pytest.fixture()
def omics_input(
    feature_dims: dict[str, int], modality_order: list[str], batch_size: int
) -> torch.Tensor:
    """Concatenated omics tensor matching feature_dims layout."""
    total = sum(feature_dims[m] for m in modality_order)
    return torch.randn(batch_size, total)


# ── Synthetic drug response data ───────────────────────────────────────────


@pytest.fixture()
def rng() -> np.random.Generator:
    """Deterministic numpy RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_response_matrix(rng: np.random.Generator) -> np.ndarray:
    """Small (5 drugs × 8 cells) response matrix with some NaN."""
    mat = rng.standard_normal((5, 8)).astype(np.float32)
    mat[0, 2] = np.nan  # sparse missing
    mat[3, 5] = np.nan
    return mat


@pytest.fixture()
def drug_names_array() -> np.ndarray:
    """Drug names matching 20 pairs (4 drugs × 5 cells each)."""
    return np.array(["DrugA"] * 5 + ["DrugB"] * 5 + ["DrugC"] * 5 + ["DrugD"] * 5)


@pytest.fixture()
def predictions_and_targets(
    rng: np.random.Generator, drug_names_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic predictions and targets for per-drug evaluation."""
    n = len(drug_names_array)
    targets = rng.standard_normal(n).astype(np.float64)
    preds = targets + rng.standard_normal(n) * 0.3  # correlated predictions
    return preds, targets
