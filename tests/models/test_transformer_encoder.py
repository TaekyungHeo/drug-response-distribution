"""Tests for TransformerEncoder model."""

import pytest
import torch

from src.models.transformer_encoder import TransformerEncoder

FEATURE_DIMS = {"rna": 50, "mutations": 30}
MODALITY_ORDER = ["rna", "mutations"]
OMICS_DIM = 80
DRUG_FP_DIM = 64
BATCH = 4
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2


@pytest.fixture
def model() -> TransformerEncoder:
    return TransformerEncoder(
        FEATURE_DIMS,
        MODALITY_ORDER,
        drug_fp_dim=DRUG_FP_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.0,
        modality_dropout_p=0.0,
    )


def _inputs(batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    x_omics = torch.randn(batch, OMICS_DIM)
    x_drug = torch.randn(batch, DRUG_FP_DIM)
    return x_omics, x_drug


def test_forward_shape(model: TransformerEncoder) -> None:
    x_omics, x_drug = _inputs()
    out = model(x_omics, x_drug)
    assert out.shape == (BATCH,)


def test_forward_no_nan(model: TransformerEncoder) -> None:
    x_omics, x_drug = _inputs()
    out = model(x_omics, x_drug)
    assert not torch.isnan(out).any()


def test_return_attention_shape(model: TransformerEncoder) -> None:
    model.eval()
    x_omics, x_drug = _inputs()
    with torch.no_grad():
        preds, attn_list = model(x_omics, x_drug, return_attention=True)
    assert preds.shape == (BATCH,)
    assert len(attn_list) == N_LAYERS
    n_tokens = len(MODALITY_ORDER) + 1  # omics + drug
    for attn in attn_list:
        assert attn.shape == (BATCH, N_HEADS, n_tokens, n_tokens)


def test_return_attention_matches_normal(model: TransformerEncoder) -> None:
    """Predictions with return_attention=True must be numerically identical."""
    model.eval()
    x_omics, x_drug = _inputs()
    with torch.no_grad():
        preds_normal = model(x_omics, x_drug, return_attention=False)
        preds_attn, _ = model(x_omics, x_drug, return_attention=True)
    assert torch.allclose(preds_normal, preds_attn, atol=1e-5)


def test_return_attention_weights_sum_to_one(model: TransformerEncoder) -> None:
    """Attention weights (softmax outputs) should sum to 1 over the key dimension."""
    model.eval()
    x_omics, x_drug = _inputs()
    with torch.no_grad():
        _, attn_list = model(x_omics, x_drug, return_attention=True)
    for attn in attn_list:
        row_sums = attn.sum(dim=-1)  # sum over keys
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_modality_dropout_training(model: TransformerEncoder) -> None:
    model_drop = TransformerEncoder(
        FEATURE_DIMS,
        MODALITY_ORDER,
        drug_fp_dim=DRUG_FP_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.0,
        modality_dropout_p=1.0,  # always drop
    )
    model_drop.train()
    x_omics, x_drug = _inputs()
    out = model_drop(x_omics, x_drug)
    assert out.shape == (BATCH,)
    assert not torch.isnan(out).any()


def test_single_sample(model: TransformerEncoder) -> None:
    out = model(*_inputs(batch=1))
    assert out.shape == (1,)
