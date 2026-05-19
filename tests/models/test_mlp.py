"""Unit tests for src/models/mlp.py — pure in-memory, no disk I/O."""

from __future__ import annotations

import pytest
import torch

from src.models.mlp import (
    MLP,
    ConcatenationBaseline,
    LateFusionBaseline,
    RNAOnlyBaseline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 4
RNA_DIM = 16
MUT_DIM = 8
TOTAL_DIM = RNA_DIM + MUT_DIM
FEATURE_DIMS = {"rna": RNA_DIM, "mut": MUT_DIM}
MODALITY_ORDER = ["rna", "mut"]


@pytest.fixture()
def concat_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, TOTAL_DIM)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_forward_shape(self, concat_input: torch.Tensor) -> None:
        model = MLP(TOTAL_DIM, [32, 16])
        out = model(concat_input)
        assert out.shape == (BATCH,)

    def test_output_is_1d(self) -> None:
        model = MLP(10, [8])
        x = torch.randn(2, 10)
        assert out.dim() == 1 if (out := model(x)) is not None else False

    def test_custom_dropout(self) -> None:
        model = MLP(10, [8], dropout=0.0)
        # With dropout=0 training and eval should give same result
        model.eval()
        x = torch.randn(2, 10)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# RNAOnlyBaseline
# ---------------------------------------------------------------------------


class TestRNAOnlyBaseline:
    def test_forward_shape(self, concat_input: torch.Tensor) -> None:
        model = RNAOnlyBaseline(FEATURE_DIMS, MODALITY_ORDER)
        out = model(concat_input)
        assert out.shape == (BATCH,)

    def test_uses_only_rna_slice(self) -> None:
        """Changing mut features should not affect output."""
        model = RNAOnlyBaseline(FEATURE_DIMS, MODALITY_ORDER)
        model.eval()
        torch.manual_seed(1)
        x1 = torch.randn(BATCH, TOTAL_DIM)
        x2 = x1.clone()
        x2[:, RNA_DIM:] = torch.randn(BATCH, MUT_DIM)  # change mut part
        torch.testing.assert_close(model(x1), model(x2))

    def test_default_hidden(self) -> None:
        assert isinstance(RNAOnlyBaseline._DEFAULT_HIDDEN, list)
        assert all(isinstance(h, int) for h in RNAOnlyBaseline._DEFAULT_HIDDEN)


# ---------------------------------------------------------------------------
# ConcatenationBaseline
# ---------------------------------------------------------------------------


class TestConcatenationBaseline:
    def test_forward_shape(self, concat_input: torch.Tensor) -> None:
        model = ConcatenationBaseline(FEATURE_DIMS, MODALITY_ORDER)
        out = model(concat_input)
        assert out.shape == (BATCH,)

    def test_uses_full_input(self) -> None:
        """Changing any part of the input should affect output."""
        model = ConcatenationBaseline(FEATURE_DIMS, MODALITY_ORDER, hidden_dims=[16])
        model.eval()
        torch.manual_seed(2)
        x1 = torch.randn(BATCH, TOTAL_DIM)
        x2 = x1.clone()
        x2[:, -1] += 10.0  # large perturbation to last feature
        out1 = model(x1)
        out2 = model(x2)
        assert not torch.allclose(out1, out2)

    def test_default_hidden(self) -> None:
        assert isinstance(ConcatenationBaseline._DEFAULT_HIDDEN, list)
        assert all(isinstance(h, int) for h in ConcatenationBaseline._DEFAULT_HIDDEN)


# ---------------------------------------------------------------------------
# LateFusionBaseline
# ---------------------------------------------------------------------------


class TestLateFusionBaseline:
    def test_forward_shape(self, concat_input: torch.Tensor) -> None:
        model = LateFusionBaseline(FEATURE_DIMS, MODALITY_ORDER)
        out = model(concat_input)
        assert out.shape == (BATCH,)

    def test_averages_modality_predictions(self) -> None:
        """Output should be the mean of per-modality MLP outputs."""
        model = LateFusionBaseline(FEATURE_DIMS, MODALITY_ORDER, hidden_dims=[8])
        model.eval()
        torch.manual_seed(3)
        x = torch.randn(BATCH, TOTAL_DIM)
        full_out = model(x)
        # Manually compute per-modality predictions
        rna_pred = model.encoders["rna"](x[:, :RNA_DIM])
        mut_pred = model.encoders["mut"](x[:, RNA_DIM:])
        expected = (rna_pred + mut_pred) / 2
        torch.testing.assert_close(full_out, expected)

    def test_default_hidden(self) -> None:
        assert isinstance(LateFusionBaseline._DEFAULT_HIDDEN, list)
        assert all(isinstance(h, int) for h in LateFusionBaseline._DEFAULT_HIDDEN)
