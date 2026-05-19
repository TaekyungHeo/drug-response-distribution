"""Generate pre-trained chemical embeddings for drugs using ChemBERTa.

from __future__ import annotations

ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) is a RoBERTa model pre-trained on
77M SMILES from ZINC. It maps SMILES strings to 768-dim embeddings that encode
rich pre-trained chemical knowledge — unlike Morgan fingerprints (fixed hashes)
or our from-scratch GCN (too few training drugs).

Embeddings are computed once and cached as drug_chembert_embeddings.npy.
"""

import json
import logging
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
D_EMBED = 768

log = logging.getLogger(__name__)


def compute_chembert_embeddings(
    smiles_dict: dict[str, str | None],
    drug_to_idx: dict[str, int],
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Generate ChemBERTa CLS embeddings for all drugs.

    Drugs with no SMILES receive zero vectors (consistent with Phase 2/3).

    Args:
        smiles_dict: Mapping drug_name -> SMILES or None.
        drug_to_idx: Mapping drug_name -> integer index.
        model_name: HuggingFace model identifier.
        batch_size: Number of SMILES to encode per forward pass.
        device: 'cpu', 'cuda', or 'mps'. Defaults to best available.

    Returns:
        float32 array of shape (n_drugs, 768).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    log.info("Loading ChemBERTa model: %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    n_drugs = len(drug_to_idx)
    embeddings = np.zeros((n_drugs, D_EMBED), dtype=np.float32)

    # Collect (idx, smiles) pairs for drugs with SMILES
    valid_pairs: list[tuple] = []
    for drug_name, idx in drug_to_idx.items():
        smiles = smiles_dict.get(drug_name)
        if smiles is not None:
            valid_pairs.append((idx, smiles))

    log.info("Encoding %d/%d drugs with SMILES", len(valid_pairs), n_drugs)

    import torch

    with torch.no_grad():
        for i in range(0, len(valid_pairs), batch_size):
            batch = valid_pairs[i : i + batch_size]
            idxs = [p[0] for p in batch]
            smiles_batch = [p[1] for p in batch]

            inputs = tokenizer(
                smiles_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # CLS token embedding (first token)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
            cls_emb_np = cls_emb.cpu().float().numpy()

            for j, drug_idx in enumerate(idxs):
                embeddings[drug_idx] = cls_emb_np[j]

            if (i // batch_size + 1) % 5 == 0:
                log.info("  encoded %d/%d", i + len(batch), len(valid_pairs))

    n_nonzero = int((embeddings.sum(axis=1) != 0).sum())
    log.info("Done. Non-zero embeddings: %d/%d", n_nonzero, n_drugs)
    return embeddings


def get_chembert_embeddings(
    drug_to_idx: dict[str, int],
    processed_dir: Path = PROCESSED_DIR,
    force_recompute: bool = False,
) -> np.ndarray:
    """Load ChemBERTa embeddings from cache or compute and save.

    Returns:
        float32 array of shape (n_drugs, 768).
    """
    emb_path = processed_dir / "drug_chembert_embeddings.npy"
    if emb_path.exists() and not force_recompute:
        log.info("Loading cached ChemBERTa embeddings from %s", emb_path)
        return np.load(emb_path)

    smiles_path = processed_dir / "drug_smiles.json"
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"drug_smiles.json not found at {smiles_path}. Run src/data/drug_features.py first."
        )
    with smiles_path.open() as f:
        smiles_dict: dict[str, str | None] = json.load(f)

    embeddings = compute_chembert_embeddings(smiles_dict, drug_to_idx)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    log.info("Saved ChemBERTa embeddings to %s", emb_path)
    return embeddings


def main() -> None:
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    drug_df = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    drugs = sorted(drug_df["drug_name"].unique())
    drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(drugs)}

    embeddings = get_chembert_embeddings(drug_to_idx, force_recompute=True)
    n_nonzero = int((embeddings.sum(axis=1) != 0).sum())
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"Non-zero rows: {n_nonzero}/{len(drug_to_idx)}")
    print(f"Embedding norm (mean): {np.linalg.norm(embeddings, axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
