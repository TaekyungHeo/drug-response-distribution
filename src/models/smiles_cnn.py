"""Multiscale CNN + Transformer encoder for SMILES strings.

from __future__ import annotations

Architecture follows PASO (Wu et al., PLoS Comp Bio 2025):
  1. Character embedding (embed_dim per character)
  2. Multiscale CNN: parallel branches with kernel_sizes [3, 5, 11]
  3. Transformer encoder (n_layers) on concatenated CNN output
  4. Mean pooling → d_out dimensional drug embedding

Input: LongTensor (batch, max_len) of character indices (0=padding)
Output: FloatTensor (batch, d_out)
"""


import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['SMILESCNNEncoder']

class MultiscaleCNN(nn.Module):
    """Parallel CNN branches with different kernel sizes, concatenated output."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 128,
        kernel_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 11]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, n_filters, k, padding=k // 2),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, in_channels, seq_len) → (batch, n_filters * n_branches, seq_len)"""
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class SMILESCNNEncoder(nn.Module):
    """Character-level SMILES encoder: embedding → multiscale CNN → transformer → pooling.

    Args:
        vocab_size: Number of tokens including padding (0=PAD, 1=UNK, 2..vocab).
        embed_dim: Character embedding dimension.
        n_filters: Number of filters per CNN branch.
        kernel_sizes: CNN kernel sizes (one branch per size).
        n_transformer_layers: Number of transformer encoder layers.
        d_out: Output embedding dimension.
        dropout: Dropout probability in transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 16,
        n_filters: int = 128,
        kernel_sizes: list[int] | None = None,
        n_transformer_layers: int = 4,
        d_out: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 11]
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        self.n_branches = len(kernel_sizes)
        cnn_out_dim = n_filters * self.n_branches  # concatenated CNN output channels

        # Character embedding (padding_idx=0 keeps padding tokens as zero vectors)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiscale CNN
        self.cnn = MultiscaleCNN(embed_dim, n_filters, kernel_sizes)

        # Project CNN output to d_out for transformer input
        self.cnn_proj = nn.Linear(cnn_out_dim, d_out)

        # Transformer encoder (optional; n_transformer_layers=0 skips it)
        self.n_transformer_layers = n_transformer_layers
        if n_transformer_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_out,
                nhead=max(1, d_out // 64),
                dim_feedforward=d_out * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module = nn.TransformerEncoder(
                enc_layer, num_layers=n_transformer_layers
            )
        else:
            self.transformer = nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: LongTensor (batch, max_len) — character indices, 0=padding.
        Returns:
            FloatTensor (batch, d_out) — drug embedding.
        """
        # Build padding mask: True where x == 0 (padding positions)
        key_padding_mask = x == 0  # (batch, max_len)

        # If ALL positions are masked (drug has no SMILES → all-zero sequence),
        # unmask the whole row to avoid NaN in transformer attention softmax.
        all_masked = key_padding_mask.all(dim=1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_masked

        # Character embedding: (batch, max_len) → (batch, max_len, embed_dim)
        emb = self.embedding(x)

        # CNN: (batch, embed_dim, max_len) → (batch, cnn_out_dim, max_len)
        cnn_out = self.cnn(emb.transpose(1, 2))

        # Project and transpose back: (batch, max_len, d_out)
        seq = self.cnn_proj(cnn_out.transpose(1, 2))

        # Transformer encoder with padding mask (skipped if n_transformer_layers=0)
        if self.n_transformer_layers > 0:
            seq = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        seq = self.norm(seq)

        # Mean pooling over non-padding positions
        mask_float = (~key_padding_mask).float().unsqueeze(-1)  # (batch, max_len, 1)
        pooled = (seq * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

        return pooled  # (batch, d_out)
