"""TransformerEncoderSMAN: TransformerEncoder + explicit drug-cell cross-attention (SMAN-inspired).

from __future__ import annotations

Architecture difference from TransformerEncoder:
  TransformerEncoder:  [omics_tokens, drug_token] → self-attention transformer → mean pool → head
  SMAN: [omics_tokens] + drug_emb → N cross-attention layers (drug↔cell) → pool → head

Cross-attention block (bidirectional):
  1. drug_emb attends to cell_tokens  (Q=drug, K=V=cell) → enriched_drug
  2. cell_tokens attend to drug_emb   (Q=cell, K=V=drug) → enriched_cell
  3. residual connections + LayerNorm + FFN on each side

After N cross-attention blocks, mean-pool enriched_cell + enriched_drug → MLP head.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.mlp import _build_offsets

__all__ = ['TransformerEncoderSMAN']

class CrossAttentionBlock(nn.Module):
    """One bidirectional cross-attention block between a drug vector and cell tokens.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Drug attends to cell tokens
        self.drug_to_cell_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_drug1 = nn.LayerNorm(d_model)
        self.norm_drug2 = nn.LayerNorm(d_model)
        self.ffn_drug = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Cell tokens attend to drug
        self.cell_to_drug_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cell1 = nn.LayerNorm(d_model)
        self.norm_cell2 = nn.LayerNorm(d_model)
        self.ffn_cell = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, drug: Tensor, cell: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            drug: (batch, 1, d_model) — drug embedding as single token.
            cell: (batch, M, d_model) — cell line omics tokens.
        Returns:
            drug: (batch, 1, d_model) — enriched drug.
            cell: (batch, M, d_model) — enriched cell tokens.
        """
        # Drug attends to cell (pre-norm)
        drug_normed = self.norm_drug1(drug)
        cell_normed_for_drug = self.norm_cell1(cell)
        drug_attn, _ = self.drug_to_cell_attn(
            query=drug_normed,
            key=cell_normed_for_drug,
            value=cell_normed_for_drug,
        )
        drug = drug + drug_attn
        drug = drug + self.ffn_drug(self.norm_drug2(drug))

        # Cell attends to drug (pre-norm)
        cell_normed = self.norm_cell1(cell)
        drug_normed_for_cell = self.norm_drug1(drug)
        cell_attn, _ = self.cell_to_drug_attn(
            query=cell_normed,
            key=drug_normed_for_cell,
            value=drug_normed_for_cell,
        )
        cell = cell + cell_attn
        cell = cell + self.ffn_cell(self.norm_cell2(cell))

        return drug, cell


class TransformerEncoderSMAN(nn.Module):
    """Per-modality encoders + drug encoder + bidirectional cross-attention → prediction.

    Args:
        feature_dims: Mapping from modality name to feature dimension.
        modality_order: Ordered list of modality names.
        drug_fp_dim: Dimension of pre-computed drug fingerprint vector.
        d_model: Hidden dimension throughout.
        n_heads: Number of attention heads in cross-attention.
        n_cross_layers: Number of bidirectional cross-attention blocks.
        n_self_layers: Number of self-attention layers on cell tokens before cross-attention.
        dropout: Dropout probability.
        modality_dropout_p: Probability of zeroing each omics token during training.
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        drug_fp_dim: int = 2048,
        d_model: int = 256,
        n_heads: int = 8,
        n_cross_layers: int = 4,
        n_self_layers: int = 2,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self._modality_order = modality_order
        self._offsets: dict[str, tuple[int, int]] = _build_offsets(feature_dims, modality_order)
        self.modality_dropout_p = modality_dropout_p

        # Per-modality linear encoders
        self.omics_encoders = nn.ModuleDict(
            {mod: nn.Linear(feature_dims[mod], d_model) for mod in modality_order}
        )

        # Drug encoder
        self.drug_encoder = nn.Linear(drug_fp_dim, d_model)

        # Learnable type embeddings (omics + drug)
        n_tokens = len(modality_order) + 1
        self.type_emb = nn.Embedding(n_tokens, d_model)

        # Optional self-attention on cell tokens before cross-attention
        if n_self_layers > 0:
            self_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.cell_self_attn: nn.Module = nn.TransformerEncoder(
                self_layer, num_layers=n_self_layers
            )
        else:
            self.cell_self_attn = nn.Identity()
        self.n_self_layers = n_self_layers

        # Cross-attention blocks
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(n_cross_layers)
            ]
        )

        # Final norm and prediction head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_omics: Tensor, x_drug_fp: Tensor) -> Tensor:
        """
        Args:
            x_omics: (batch, total_omics_features)
            x_drug_fp: (batch, drug_fp_dim)
        Returns:
            predictions: (batch,) scalar LN_IC50.
        """
        # Build cell token sequence
        cell_tokens: list[Tensor] = []
        for i, mod in enumerate(self._modality_order):
            start, end = self._offsets[mod]
            x_mod = x_omics[:, start:end]
            if self.training and self.modality_dropout_p > 0:
                mask = (
                    torch.rand(x_mod.shape[0], 1, device=x_mod.device) > self.modality_dropout_p
                ).float()
                x_mod = x_mod * mask
            tok = self.omics_encoders[mod](x_mod) + self.type_emb.weight[i]
            cell_tokens.append(tok.unsqueeze(1))
        cell = torch.cat(cell_tokens, dim=1)  # (batch, M, d_model)

        # Self-attention over cell tokens
        if self.n_self_layers > 0:
            cell = self.cell_self_attn(cell)

        # Drug embedding
        drug = (
            self.drug_encoder(x_drug_fp) + self.type_emb.weight[len(self._modality_order)]
        ).unsqueeze(1)  # (batch, 1, d_model)

        # Bidirectional cross-attention blocks
        for block in self.cross_blocks:
            drug, cell = block(drug, cell)

        # Pool: mean over all tokens (cell + drug)
        all_tokens = torch.cat([cell, drug], dim=1)  # (batch, M+1, d_model)
        pooled = self.norm(all_tokens.mean(dim=1))
        return self.head(pooled).squeeze(-1)
