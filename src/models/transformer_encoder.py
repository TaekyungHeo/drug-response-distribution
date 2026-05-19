"""TransformerEncoder v1: cross-attention transformer over omics + drug tokens."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from src.models.mlp import _build_offsets

__all__ = ['TransformerEncoder']

class TransformerEncoder(nn.Module):
    """Per-modality linear encoders + drug token → TransformerEncoder → mean pool → prediction.

    Args:
        feature_dims: Mapping from modality name to feature dimension.
        modality_order: Ordered list of modality names (must match feature_dims keys).
        drug_fp_dim: Dimension of pre-computed drug fingerprint vector.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of TransformerEncoderLayer stacks.
        dropout: Dropout probability inside transformer layers.
        modality_dropout_p: Probability of zeroing each omics token (training only).
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        drug_fp_dim: int = 2048,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
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

        # Drug encoder (operates on pre-computed fingerprint)
        self.drug_encoder = nn.Linear(drug_fp_dim, d_model)

        # Learnable type embeddings: one per omics modality + one for drug
        n_tokens = len(modality_order) + 1
        self.type_emb = nn.Embedding(n_tokens, d_model)

        # Transformer encoder with pre-norm (more stable)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Prediction head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _build_token_sequence(self, x_omics: Tensor, x_drug_fp: Tensor) -> Tensor:
        """Build the (batch, n_tokens, d_model) token sequence."""
        tokens: list[Tensor] = []

        for i, mod in enumerate(self._modality_order):
            start, end = self._offsets[mod]
            x_mod = x_omics[:, start:end]

            # Modality dropout: zero entire token per sample independently (training only)
            if self.training and self.modality_dropout_p > 0:
                mask = (
                    torch.rand(x_mod.shape[0], 1, device=x_mod.device) > self.modality_dropout_p
                ).float()
                x_mod = x_mod * mask

            tok = self.omics_encoders[mod](x_mod) + self.type_emb.weight[i]
            tokens.append(tok.unsqueeze(1))

        drug_tok = self.drug_encoder(x_drug_fp) + self.type_emb.weight[len(self._modality_order)]
        tokens.append(drug_tok.unsqueeze(1))

        return torch.cat(tokens, dim=1)  # (batch, n_tokens, d_model)

    def forward(
        self,
        x_omics: Tensor,
        x_drug_fp: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """
        Args:
            x_omics: (batch, total_omics_features) — same concat format as Phase 1 models.
            x_drug_fp: (batch, drug_fp_dim) — pre-computed Morgan fingerprint (float32 {0,1}).
            return_attention: If True, also return per-layer attention weights.
        Returns:
            predictions: (batch,) scalar LN_IC50.
            attn_weights (only if return_attention=True): list of length n_layers,
                each tensor (batch, n_heads, n_tokens, n_tokens).
        """
        seq = self._build_token_sequence(x_omics, x_drug_fp)

        if not return_attention:
            seq = self.transformer(seq)
        else:
            attn_list: list[Tensor] = []
            for layer in self.transformer.layers:
                enc_layer = cast(nn.TransformerEncoderLayer, layer)
                # Pre-norm self-attention (norm_first=True)
                normed = enc_layer.norm1(seq)
                attn_out, attn_weights = enc_layer.self_attn(
                    normed,
                    normed,
                    normed,
                    need_weights=True,
                    average_attn_weights=False,
                )
                # attn_weights: (batch, n_heads, n_tokens, n_tokens)
                attn_list.append(attn_weights.detach())
                seq = seq + enc_layer.dropout1(attn_out)
                # Pre-norm feedforward
                seq = seq + enc_layer.dropout2(
                    enc_layer.linear2(
                        enc_layer.dropout(
                            enc_layer.activation(enc_layer.linear1(enc_layer.norm2(seq)))
                        )
                    )
                )

        pooled = seq.mean(dim=1)
        pooled = self.norm(pooled)
        preds = self.head(pooled).squeeze(-1)

        if return_attention:
            return preds, attn_list
        return preds
