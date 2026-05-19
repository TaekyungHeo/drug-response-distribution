"""TransformerEncoderSmiles: TransformerEncoder with SMILES CNN drug encoder instead of Morgan FP.

from __future__ import annotations

Identical to TransformerEncoder except the drug representation is learned end-to-end
from SMILES character sequences via a multiscale CNN + transformer encoder.

Drug input: LongTensor (batch, max_len) of character indices (0=padding).
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.mlp import _build_offsets
from src.models.smiles_cnn import SMILESCNNEncoder

__all__ = ['TransformerEncoderSmiles']

class TransformerEncoderSmiles(nn.Module):
    """Per-modality linear encoders + SMILES CNN drug encoder → Transformer → prediction.

    Args:
        feature_dims: Mapping from modality name to feature dimension.
        modality_order: Ordered list of modality names.
        vocab_size: Number of SMILES character tokens (including padding).
        smiles_max_len: Maximum SMILES sequence length.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads in the main transformer.
        n_layers: Number of main transformer encoder layers.
        dropout: Dropout probability.
        modality_dropout_p: Probability of zeroing each omics token during training.
        smiles_embed_dim: Character embedding dim in SMILES encoder.
        smiles_n_filters: Filters per CNN branch.
        smiles_n_layers: Transformer layers in SMILES encoder.
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        vocab_size: int,
        smiles_max_len: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.3,
        smiles_embed_dim: int = 16,
        smiles_n_filters: int = 128,
        smiles_n_layers: int = 4,
    ) -> None:
        super().__init__()
        self._modality_order = modality_order
        self._offsets = _build_offsets(feature_dims, modality_order)
        self.modality_dropout_p = modality_dropout_p

        # Per-modality linear encoders (same as TransformerEncoder)
        self.omics_encoders = nn.ModuleDict(
            {mod: nn.Linear(feature_dims[mod], d_model) for mod in modality_order}
        )

        # SMILES CNN drug encoder (replaces nn.Linear(drug_fp_dim, d_model))
        self.drug_encoder = SMILESCNNEncoder(
            vocab_size=vocab_size,
            embed_dim=smiles_embed_dim,
            n_filters=smiles_n_filters,
            kernel_sizes=[3, 5, 11],
            n_transformer_layers=smiles_n_layers,
            d_out=d_model,
            dropout=dropout,
        )

        # Learnable type embeddings
        n_tokens = len(modality_order) + 1
        self.type_emb = nn.Embedding(n_tokens, d_model)

        # Main transformer encoder
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

    def forward(self, x_omics: Tensor, x_drug_smiles: Tensor) -> Tensor:
        """
        Args:
            x_omics: (batch, total_omics_features)
            x_drug_smiles: LongTensor (batch, max_len) — SMILES character indices.
        Returns:
            predictions: (batch,) scalar LN_IC50.
        """
        tokens: list[Tensor] = []

        for i, mod in enumerate(self._modality_order):
            start, end = self._offsets[mod]
            x_mod = x_omics[:, start:end]
            if self.training and self.modality_dropout_p > 0:
                mask = (
                    torch.rand(x_mod.shape[0], 1, device=x_mod.device) > self.modality_dropout_p
                ).float()
                x_mod = x_mod * mask
            tok = self.omics_encoders[mod](x_mod) + self.type_emb.weight[i]
            tokens.append(tok.unsqueeze(1))

        # Drug token from SMILES CNN encoder
        drug_emb = self.drug_encoder(x_drug_smiles)  # (batch, d_model)
        drug_tok = drug_emb + self.type_emb.weight[len(self._modality_order)]
        tokens.append(drug_tok.unsqueeze(1))

        seq = torch.cat(tokens, dim=1)  # (batch, n_tokens, d_model)
        seq = self.transformer(seq)
        pooled = self.norm(seq.mean(dim=1))
        return self.head(pooled).squeeze(-1)
