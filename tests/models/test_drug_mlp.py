"""Tests for DrugFingerprintMLP and DrugAwareMLP."""

import pytest
import torch

from src.models.drug_mlp import DrugAwareMLP, DrugFingerprintMLP

FEATURE_DIMS = {"rna": 50, "mutations": 30}
MODALITY_ORDER = ["rna", "mutations"]
OMICS_DIM = 80
BATCH = 4


@pytest.fixture
def fp_mlp() -> DrugFingerprintMLP:
    return DrugFingerprintMLP(
        FEATURE_DIMS, MODALITY_ORDER, drug_fp_dim=64, drug_emb_dim=16, hidden_dims=[32, 16]
    )


@pytest.fixture
def aware_mlp() -> DrugAwareMLP:
    return DrugAwareMLP(
        FEATURE_DIMS, MODALITY_ORDER, n_drugs=10, drug_emb_dim=16, hidden_dims=[32, 16]
    )


def test_fp_mlp_output_shape(fp_mlp: DrugFingerprintMLP) -> None:
    x_omics = torch.randn(BATCH, OMICS_DIM)
    x_drug = torch.randn(BATCH, 64)
    out = fp_mlp(x_omics, x_drug)
    assert out.shape == (BATCH,)


def test_fp_mlp_no_nan(fp_mlp: DrugFingerprintMLP) -> None:
    x_omics = torch.randn(BATCH, OMICS_DIM)
    x_drug = torch.randn(BATCH, 64)
    out = fp_mlp(x_omics, x_drug)
    assert not torch.isnan(out).any()


def test_fp_mlp_binary_input(fp_mlp: DrugFingerprintMLP) -> None:
    x_omics = torch.randn(BATCH, OMICS_DIM)
    x_drug = torch.zeros(BATCH, 64)
    x_drug[:, :10] = 1.0  # sparse binary fingerprint
    out = fp_mlp(x_omics, x_drug)
    assert out.shape == (BATCH,)
    assert not torch.isnan(out).any()


def test_aware_mlp_output_shape(aware_mlp: DrugAwareMLP) -> None:
    x_omics = torch.randn(BATCH, OMICS_DIM)
    drug_idx = torch.randint(0, 10, (BATCH,))
    out = aware_mlp(x_omics, drug_idx)
    assert out.shape == (BATCH,)


def test_aware_mlp_no_nan(aware_mlp: DrugAwareMLP) -> None:
    x_omics = torch.randn(BATCH, OMICS_DIM)
    drug_idx = torch.randint(0, 10, (BATCH,))
    out = aware_mlp(x_omics, drug_idx)
    assert not torch.isnan(out).any()


def test_fp_mlp_single_sample(fp_mlp: DrugFingerprintMLP) -> None:
    fp_mlp.eval()  # BatchNorm requires batch>1 in training mode
    with torch.no_grad():
        out = fp_mlp(torch.randn(1, OMICS_DIM), torch.randn(1, 64))
    assert out.shape == (1,)
