"""TransformerEncoder multi-task variant: shared encoder + two prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.mlp import _build_offsets

__all__ = ['TransformerEncoderMultiTask']

class TransformerEncoderMultiTask(nn.Module):
    """Shared TransformerEncoder encoder + task-specific heads for GDSC2 (IC50) and CTRPv2 (AUC).

    The full transformer backbone is shared; only the final linear head differs per task.
    Both tasks use Morgan FP drug encoding and RNA+mutations cell encoding.

    Args:
        feature_dims: Mapping from modality name to feature dimension.
        modality_order: Ordered list of modality names.
        drug_fp_dim: Dimension of Morgan fingerprint vector.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of TransformerEncoderLayer stacks.
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
        n_layers: int = 4,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self._modality_order = modality_order
        self._offsets: dict[str, tuple[int, int]] = _build_offsets(feature_dims, modality_order)
        self.modality_dropout_p = modality_dropout_p

        self.omics_encoders = nn.ModuleDict(
            {mod: nn.Linear(feature_dims[mod], d_model) for mod in modality_order}
        )
        self.drug_encoder = nn.Linear(drug_fp_dim, d_model)

        n_tokens = len(modality_order) + 1
        self.type_emb = nn.Embedding(n_tokens, d_model)

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
        # Task A: GDSC2 LN(IC50)
        self.head_ic50 = nn.Linear(d_model, 1)
        # Task B: CTRPv2 AUC
        self.head_auc = nn.Linear(d_model, 1)

    def _encode(self, x_omics: Tensor, x_drug_fp: Tensor) -> Tensor:
        """Shared encoder: omics + drug → pooled representation (batch, d_model)."""
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
        drug_tok = self.drug_encoder(x_drug_fp) + self.type_emb.weight[len(self._modality_order)]
        tokens.append(drug_tok.unsqueeze(1))
        seq = torch.cat(tokens, dim=1)
        seq = self.transformer(seq)
        pooled = self.norm(seq.mean(dim=1))
        return pooled

    def forward(self, x_omics: Tensor, x_drug_fp: Tensor, task: str = "gdsc2") -> Tensor:
        """
        Args:
            x_omics: (batch, total_omics_features)
            x_drug_fp: (batch, drug_fp_dim)
            task: "gdsc2" → IC50 head, "ctrpv2" → AUC head
        Returns:
            predictions: (batch,)
        """
        pooled = self._encode(x_omics, x_drug_fp)
        if task == "gdsc2":
            return self.head_ic50(pooled).squeeze(-1)
        return self.head_auc(pooled).squeeze(-1)

    def forward_gdsc2(self, x_omics: Tensor, x_drug_fp: Tensor) -> Tensor:
        return self.forward(x_omics, x_drug_fp, task="gdsc2")

    def forward_ctrpv2(self, x_omics: Tensor, x_drug_fp: Tensor) -> Tensor:
        return self.forward(x_omics, x_drug_fp, task="ctrpv2")
