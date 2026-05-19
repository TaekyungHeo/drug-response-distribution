"""Drug-aware MLP: omics concat + drug embedding → MLP prediction."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from torch import Tensor

from src.models.mlp import MLP


class DrugAwareMLP(nn.Module):
    """Omics concat + learned drug embedding → MLP. Drug identity via embedding lookup."""

    _DEFAULT_HIDDEN: ClassVar[list[int]] = [1024, 512, 128]

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        n_drugs: int,
        drug_emb_dim: int = 64,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = self._DEFAULT_HIDDEN
        omics_dim = sum(feature_dims[m] for m in modality_order)
        self.drug_emb = nn.Embedding(n_drugs, drug_emb_dim)
        self.mlp = MLP(omics_dim + drug_emb_dim, hidden_dims)

    def forward(self, x_omics: Tensor, drug_idx: Tensor) -> Tensor:
        drug_vec = self.drug_emb(drug_idx)  # (batch, drug_emb_dim)
        x = torch.cat([x_omics, drug_vec], dim=1)
        return self.mlp(x)


class DrugFingerprintMLP(nn.Module):
    """Omics concat + pre-computed drug fingerprint → MLP. Supports drug-blind generalization."""

    _DEFAULT_HIDDEN: ClassVar[list[int]] = [1024, 512, 128]

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        drug_fp_dim: int = 2048,
        drug_emb_dim: int = 64,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = self._DEFAULT_HIDDEN
        omics_dim = sum(feature_dims[m] for m in modality_order)
        self.drug_emb = nn.Linear(drug_fp_dim, drug_emb_dim)
        self.mlp = MLP(omics_dim + drug_emb_dim, hidden_dims)

    def forward(self, x_omics: Tensor, drug_fp: Tensor) -> Tensor:
        drug_vec = self.drug_emb(drug_fp)
        x = torch.cat([x_omics, drug_vec], dim=1)
        return self.mlp(x)


__all__ = ["DrugAwareMLP", "DrugFingerprintMLP"]
