"""TransformerEncoderGNN: cross-attention transformer with GNN drug encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.drug_gnn import MolecularGCN
from src.models.mlp import _build_offsets

__all__ = ['TransformerEncoderGNN']

class TransformerEncoderGNN(nn.Module):
    """Per-modality encoders + GNN drug encoder → TransformerEncoder → mean pool.

    Drug graphs are stored as registered buffers and move to device automatically
    with model.to(device). The GNN runs only on unique drugs per batch
    (deduplication via torch.unique), reducing compute by ~7x at batch_size=2048.

    Interface: forward(x_omics, drug_idx) — takes integer drug indices.
    Compatible with train_drug() and predict_drug() from trainer_drug.py.

    Args:
        feature_dims:     Mapping from modality name to feature dimension.
        modality_order:   Ordered list of modality names.
        drug_atom_feats:  (n_drugs, max_atoms, d_atom) float32 tensor.
        drug_adj:         (n_drugs, max_atoms, max_atoms) float32 normalised adjacency.
        drug_mask:        (n_drugs, max_atoms) bool — True for real atoms.
        d_model:          Transformer hidden dimension.
        n_heads:          Number of attention heads.
        n_layers:         Number of transformer layers.
        gnn_layers:       Number of GCN layers in the drug encoder.
        dropout:          Dropout inside transformer and GCN hidden layers.
        modality_dropout_p: Probability of zeroing each omics token per sample (training).
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        modality_order: list[str],
        drug_atom_feats: Tensor,
        drug_adj: Tensor,
        drug_mask: Tensor,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        gnn_layers: int = 3,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self._modality_order = modality_order
        self._offsets: dict[str, tuple[int, int]] = _build_offsets(feature_dims, modality_order)
        self.modality_dropout_p = modality_dropout_p

        # Drug graph buffers: auto-move to device with model.to()
        self.register_buffer("drug_atom_feats", drug_atom_feats)  # (n, max_atoms, d_atom)
        self.register_buffer("drug_adj", drug_adj)  # (n, max_atoms, max_atoms)
        self.register_buffer("drug_mask", drug_mask)  # (n, max_atoms) bool

        d_atom = drug_atom_feats.shape[2]

        # GNN drug encoder: d_atom → d_model
        self.drug_gnn = MolecularGCN(
            d_atom=d_atom,
            d_hidden=d_model,
            d_out=d_model,
            n_layers=gnn_layers,
            dropout=dropout,
        )

        # Per-modality linear encoders
        self.omics_encoders = nn.ModuleDict(
            {mod: nn.Linear(feature_dims[mod], d_model) for mod in modality_order}
        )

        # Learnable type embeddings: one per omics modality + one for drug
        n_tokens = len(modality_order) + 1
        self.type_emb = nn.Embedding(n_tokens, d_model)

        # Transformer encoder with pre-norm (more stable training)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _encode_drugs(self, drug_idx: Tensor) -> Tensor:
        """Run GNN only on unique drugs, then expand to batch order.

        Reduces GNN compute from batch_size (~2048) to n_unique (~286).
        """
        unique_idx, inverse = torch.unique(drug_idx, return_inverse=True)

        atom_feats = self.drug_atom_feats[unique_idx]  # (n_unique, max_atoms, d_atom)
        adj = self.drug_adj[unique_idx]  # (n_unique, max_atoms, max_atoms)
        mask = self.drug_mask[unique_idx]  # (n_unique, max_atoms) bool

        drug_embs = self.drug_gnn(atom_feats, adj, mask)  # (n_unique, d_model)
        return drug_embs[inverse]  # (batch, d_model)

    def forward(self, x_omics: Tensor, drug_idx: Tensor) -> Tensor:
        """
        Args:
            x_omics:   (batch, total_omics_features) — same concat format as Phase 1/2.
            drug_idx:  (batch,) int64 — drug indices into registered graph buffers.
        Returns:
            (batch,) — predicted ln(IC50) values.
        """
        tokens: list[Tensor] = []

        # Omics tokens with optional modality dropout
        for i, mod in enumerate(self._modality_order):
            start, end = self._offsets[mod]
            x_mod = x_omics[:, start:end]

            if self.training and self.modality_dropout_p > 0:
                drop_mask = (
                    torch.rand(x_mod.shape[0], 1, device=x_mod.device) > self.modality_dropout_p
                ).float()
                x_mod = x_mod * drop_mask

            tok = self.omics_encoders[mod](x_mod) + self.type_emb.weight[i]
            tokens.append(tok.unsqueeze(1))

        # Drug token via GNN
        drug_emb = self._encode_drugs(drug_idx)
        drug_tok = drug_emb + self.type_emb.weight[len(self._modality_order)]
        tokens.append(drug_tok.unsqueeze(1))

        seq = torch.cat(tokens, dim=1)  # (batch, n_tokens, d_model)
        seq = self.transformer(seq)
        pooled = self.norm(seq.mean(dim=1))
        return self.head(pooled).squeeze(-1)
