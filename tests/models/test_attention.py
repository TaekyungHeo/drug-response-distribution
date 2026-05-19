"""Tests for attention map extraction utilities."""

import numpy as np
import pytest

from src.analysis.attention import attention_heatmap_data, mean_attention_to_drug

TOKEN_NAMES = ["rna", "mutations", "cnv", "drug"]
N_PAIRS, N_LAYERS, N_HEADS, N_TOKENS = 8, 4, 8, 4


@pytest.fixture
def dummy_attn() -> np.ndarray:
    """Random attention weights that sum to 1 over the key dimension."""
    rng = np.random.default_rng(42)
    raw = rng.random((N_PAIRS, N_LAYERS, N_HEADS, N_TOKENS, N_TOKENS)).astype(np.float32)
    # Normalize so each row sums to 1 (softmax-like)
    raw /= raw.sum(axis=-1, keepdims=True)
    return raw


def test_mean_attention_to_drug_shape(dummy_attn: np.ndarray) -> None:
    result = mean_attention_to_drug(dummy_attn, TOKEN_NAMES)
    n_omics = len([t for t in TOKEN_NAMES if t != "drug"])
    assert result.shape == (n_omics,)


def test_mean_attention_to_drug_nonnegative(dummy_attn: np.ndarray) -> None:
    result = mean_attention_to_drug(dummy_attn, TOKEN_NAMES)
    assert (result >= 0).all()


def test_heatmap_mean_shape(dummy_attn: np.ndarray) -> None:
    heatmap = attention_heatmap_data(dummy_attn, TOKEN_NAMES, aggregation="mean")
    assert heatmap.shape == (N_TOKENS, N_TOKENS)


def test_heatmap_last_layer_shape(dummy_attn: np.ndarray) -> None:
    heatmap = attention_heatmap_data(dummy_attn, TOKEN_NAMES, aggregation="last_layer")
    assert heatmap.shape == (N_TOKENS, N_TOKENS)


def test_heatmap_rows_sum_to_one(dummy_attn: np.ndarray) -> None:
    """Mean heatmap rows should sum to 1 since attention weights are normalized."""
    heatmap = attention_heatmap_data(dummy_attn, TOKEN_NAMES, aggregation="mean")
    row_sums = heatmap.sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(N_TOKENS), atol=1e-5)


def test_heatmap_invalid_aggregation(dummy_attn: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unknown aggregation"):
        attention_heatmap_data(dummy_attn, TOKEN_NAMES, aggregation="invalid")


def test_mean_attention_drug_excluded(dummy_attn: np.ndarray) -> None:
    result = mean_attention_to_drug(dummy_attn, TOKEN_NAMES)
    # Result length = n_omics (drug token not in output)
    assert len(result) == len(TOKEN_NAMES) - 1
