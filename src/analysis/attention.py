"""Attention map extraction and analysis for TransformerEncoder."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

__all__ = ['extract_attention_maps']

if TYPE_CHECKING:
    from src.data.dataset import MultiOmicsDataset
    from src.models.transformer_encoder import TransformerEncoder


def extract_attention_maps(
    model: TransformerEncoder,
    dataset: MultiOmicsDataset,
    pair_indices: np.ndarray,
    fp_matrix: np.ndarray,
    device: str = "mps",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Extract per-layer attention weights for a set of (cell, drug) pairs.

    Args:
        model: Trained TransformerEncoder in eval mode.
        dataset: MultiOmicsDataset (same one used for training).
        pair_indices: 1-D int array of pair indices to extract.
        fp_matrix: (n_drugs, 2048) float32 drug fingerprint matrix.
        device: Torch device string.
        batch_size: Pairs processed per batch (256 is safe on MPS).

    Returns:
        dict with keys:
            "attn_weights": np.ndarray (n_pairs, n_layers, n_heads, n_tokens, n_tokens)
            "pair_indices": np.ndarray (n_pairs,)
            "token_names": list[str] — omics modality names + ["drug"]
    """
    model.eval()

    token_names: list[str] = [*list(dataset.omics_to_use), "drug"]
    n_pairs = len(pair_indices)
    n_layers = len(model.transformer.layers)
    first_layer = cast(torch.nn.TransformerEncoderLayer, model.transformer.layers[0])
    n_heads = int(first_layer.self_attn.num_heads)
    n_tokens = len(token_names)

    all_attn = np.zeros((n_pairs, n_layers, n_heads, n_tokens, n_tokens), dtype=np.float32)

    # Build concat omics matrix once (same logic as _DrugFpPrefetcher)
    omics_parts = [dataset.omics_arrays[m] for m in dataset.omics_to_use]
    concat_np = np.concatenate(omics_parts, axis=1)  # (n_cells, concat_dim)

    with torch.no_grad():
        for start in range(0, n_pairs, batch_size):
            batch_idx = pair_indices[start : start + batch_size]
            cell_rows = dataset._cell_rows[batch_idx]
            drug_idxs = dataset._drug_idxs[batch_idx]

            x_np = concat_np[cell_rows]
            fp_np = fp_matrix[drug_idxs]

            x = torch.from_numpy(x_np).to(device)
            x_drug = torch.from_numpy(fp_np).to(device)

            _, attn_list = model.forward(x, x_drug, return_attention=True)

            for layer_i, attn in enumerate(attn_list):
                all_attn[start : start + len(batch_idx), layer_i] = attn.cpu().numpy()

    return {
        "attn_weights": all_attn,
        "pair_indices": pair_indices,
        "token_names": token_names,
    }


def mean_attention_to_drug(attn_weights: np.ndarray, token_names: list[str]) -> np.ndarray:
    """Mean attention from each omics token to the drug token, averaged across pairs/layers/heads.

    Args:
        attn_weights: (n_pairs, n_layers, n_heads, n_tokens, n_tokens)
        token_names: list of token names; "drug" must be one of them.

    Returns:
        (n_omics_tokens,) float32 array, ordered as omics tokens (drug token excluded).
    """
    drug_idx = token_names.index("drug")
    # attn_weights[..., query, key]: attention from query to key
    # We want: for each omics token as query, how much does it attend to drug as key?
    omics_indices = [i for i, n in enumerate(token_names) if n != "drug"]
    # Mean over pairs (0), layers (1), heads (2)
    mean_attn = attn_weights.mean(axis=(0, 1, 2))  # (n_tokens, n_tokens)
    return mean_attn[omics_indices, drug_idx]


def attention_heatmap_data(
    attn_weights: np.ndarray,
    token_names: list[str],
    aggregation: str = "mean",
) -> np.ndarray:
    """Aggregate attention weights to a (n_tokens, n_tokens) heatmap matrix.

    Args:
        attn_weights: (n_pairs, n_layers, n_heads, n_tokens, n_tokens)
        token_names: list of token names.
        aggregation: "mean" — average over pairs, layers, heads;
                     "last_layer" — last layer only, average over pairs and heads.

    Returns:
        (n_tokens, n_tokens) float32 array.
    """
    if aggregation == "mean":
        return attn_weights.mean(axis=(0, 1, 2))
    elif aggregation == "last_layer":
        return attn_weights[:, -1, :, :, :].mean(axis=(0, 1))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}. Use 'mean' or 'last_layer'.")


def save_attention_data(data: dict[str, Any], save_dir: Path) -> None:
    """Save attention map data to save_dir as a .npz file."""
    import json

    save_dir.mkdir(parents=True, exist_ok=True)
    attn: np.ndarray = data["attn_weights"]
    idx: np.ndarray = data["pair_indices"]
    np.savez_compressed(save_dir / "attention_maps.npz", attn_weights=attn, pair_indices=idx)
    (save_dir / "token_names.json").write_text(json.dumps(data["token_names"]))


def load_attention_data(save_dir: Path) -> dict[str, Any]:
    """Load attention map data from save_dir."""
    import json

    npz = np.load(save_dir / "attention_maps.npz")
    token_names = json.loads((save_dir / "token_names.json").read_text())
    return {
        "attn_weights": npz["attn_weights"],
        "pair_indices": npz["pair_indices"],
        "token_names": token_names,
    }
