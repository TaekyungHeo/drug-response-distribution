"""Molecular GCN for drug encoding: padded dense GCN compatible with MPS/CUDA/CPU."""

import torch
import torch.nn as nn
from torch import Tensor

from src.data.drug_graph import D_ATOM

__all__ = ['MolecularGCN']

class GCNLayer(nn.Module):
    """Single GCN layer: H_new = ReLU(A_hat @ H @ W).

    Uses torch.bmm for batched sparse-dense multiply — MPS-compatible.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            h:   (batch, max_atoms, d_in)  — atom feature matrix
            adj: (batch, max_atoms, max_atoms) — pre-normalised adjacency
        Returns:
            (batch, max_atoms, d_out)
        """
        h_agg = torch.bmm(adj, h)  # (batch, max_atoms, d_in)
        return torch.relu(self.dropout(self.linear(h_agg)))


class MolecularGCN(nn.Module):
    """Multi-layer GCN with masked mean pooling for molecular graphs.

    Design choices:
    - Pure PyTorch (no torch_geometric) — fully MPS-compatible
    - Padded dense adjacency — allows batched bmm on GPU/MPS
    - Pre-normalised adjacency (D^{-0.5} A_tilde D^{-0.5}) stored with graphs,
      not recomputed at runtime
    - Masked mean pooling: ignores zero-padded atoms

    Args:
        d_atom:    Input atom feature dimension (default 74).
        d_hidden:  Hidden layer dimension.
        d_out:     Output drug embedding dimension.
        n_layers:  Number of GCN layers (default 3).
        dropout:   Dropout applied between hidden layers (not after last).
    """

    def __init__(
        self,
        d_atom: int = D_ATOM,
        d_hidden: int = 256,
        d_out: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims: list[int] = [d_atom] + [d_hidden] * (n_layers - 1) + [d_out]
        self.layers = nn.ModuleList(
            [
                GCNLayer(dims[i], dims[i + 1], dropout=dropout if i < n_layers - 1 else 0.0)
                for i in range(n_layers)
            ]
        )

    def forward(self, atom_feats: Tensor, adj: Tensor, mask: Tensor) -> Tensor:
        """Run GCN and return pooled drug embeddings.

        Args:
            atom_feats: (batch, max_atoms, d_atom)
            adj:        (batch, max_atoms, max_atoms) — pre-normalised
            mask:       (batch, max_atoms) — True for real atoms
        Returns:
            (batch, d_out) — masked-mean-pooled embeddings
        """
        h = atom_feats
        for layer in self.layers:
            h = layer(h, adj)

        # Masked mean pool: zero out padding before summing
        mask_f = mask.unsqueeze(-1).float()  # (batch, max_atoms, 1)
        h = h * mask_f  # zero padding contributions
        count = mask_f.sum(dim=1).clamp(min=1.0)  # (batch, 1)
        return h.sum(dim=1) / count  # (batch, d_out)
