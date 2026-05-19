"""MLP baseline models for drug response prediction.

All models receive a pre-concatenated float32 tensor (batch × total_features).
Feature slicing per modality is done via modality_offsets computed at init.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['MLP', 'ConcatenationBaseline', 'LateFusionBaseline', 'RNAOnlyBaseline']

class MLP(nn.Module):
    """Generic multi-layer perceptron with BatchNorm, ReLU, and Dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def _build_offsets(feature_dims: dict[str, int], modality_order: list[str]) -> dict[str, tuple]:
    """Compute (start, end) slice offsets for each modality in the concat tensor."""
    offsets = {}
    pos = 0
    for mod in modality_order:
        dim = feature_dims[mod]
        offsets[mod] = (pos, pos + dim)
        pos += dim
    return offsets


class RNAOnlyBaseline(nn.Module):
    """MLP trained on RNA-seq features only (first slice of concat tensor)."""

    _DEFAULT_HIDDEN: ClassVar[list[int]] = [512, 256, 64]

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = self._DEFAULT_HIDDEN
        self._offsets = _build_offsets(feature_dims, modality_order)
        rna_start, rna_end = self._offsets["rna"]
        self._rna_start = rna_start
        self._rna_end = rna_end
        self.mlp = MLP(rna_end - rna_start, hidden_dims)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x[:, self._rna_start : self._rna_end])


class ConcatenationBaseline(nn.Module):
    """MLP trained on all modalities concatenated (uses full concat tensor directly)."""

    _DEFAULT_HIDDEN: ClassVar[list[int]] = [1024, 512, 128]

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = self._DEFAULT_HIDDEN
        total_dim = sum(feature_dims[m] for m in modality_order)
        self.mlp = MLP(total_dim, hidden_dims)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class LateFusionBaseline(nn.Module):
    """Separate MLP per modality, predictions averaged."""

    _DEFAULT_HIDDEN: ClassVar[list[int]] = [256, 64]

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = self._DEFAULT_HIDDEN
        self._offsets = _build_offsets(feature_dims, modality_order)
        self._modality_order = modality_order
        self.encoders = nn.ModuleDict(
            {mod: MLP(feature_dims[mod], hidden_dims) for mod in modality_order}
        )

    def forward(self, x: Tensor) -> Tensor:
        preds = torch.stack(
            [self.encoders[mod](x[:, s:e]) for mod, (s, e) in self._offsets.items()],
            dim=1,
        )
        return preds.mean(dim=1)
