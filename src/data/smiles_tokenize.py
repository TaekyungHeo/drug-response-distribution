"""Character-level SMILES tokenizer for drug encoding.

Builds a character vocabulary from drug SMILES strings and encodes each drug
as a fixed-length integer sequence (padded/truncated).

Output: (n_drugs, max_len) int32 array, 0=padding, 1=unknown, 2..vocab=characters.
"""

import json
import logging
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MAX_LEN = 128  # covers 95th percentile of SMILES lengths in GDSC2

log = logging.getLogger(__name__)


def build_vocab(smiles_list: list[str]) -> dict[str, int]:
    """Build character vocabulary from SMILES strings.

    Returns:
        char → index mapping. Index 0 reserved for padding, 1 for unknown.
        Actual characters start at index 2.
    """
    chars = set()
    for s in smiles_list:
        if s:
            chars.update(s)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, c in enumerate(sorted(chars), start=2):
        vocab[c] = i
    return vocab


def encode_smiles(
    smiles: str | None,
    vocab: dict[str, int],
    max_len: int = MAX_LEN,
) -> np.ndarray:
    """Encode a SMILES string to a fixed-length integer array.

    Truncates to max_len; pads shorter sequences with 0.
    Drugs without SMILES return all-zeros (padding token = no drug info).
    """
    arr = np.zeros(max_len, dtype=np.int32)
    if not smiles:
        return arr
    unk = vocab["<UNK>"]
    for i, c in enumerate(smiles[:max_len]):
        arr[i] = vocab.get(c, unk)
    return arr


def build_smiles_matrix(
    drug_to_idx: dict[str, int],
    smiles_dict: dict[str, str | None],
    vocab: dict[str, int] | None = None,
    max_len: int = MAX_LEN,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build (n_drugs, max_len) integer matrix of SMILES encodings.

    Args:
        drug_to_idx: drug_name → row index.
        smiles_dict: drug_name → SMILES string or None.
        vocab: Pre-built vocabulary. If None, builds from smiles_dict values.
        max_len: Sequence length (pad/truncate).

    Returns:
        matrix: int32 array (n_drugs, max_len).
        vocab: character → index mapping.
    """
    if vocab is None:
        valid_smiles = [v for v in smiles_dict.values() if v]
        vocab = build_vocab(valid_smiles)

    n_drugs = len(drug_to_idx)
    matrix = np.zeros((n_drugs, max_len), dtype=np.int32)
    n_covered = 0
    for drug_name, idx in drug_to_idx.items():
        smiles = smiles_dict.get(drug_name)
        matrix[idx] = encode_smiles(smiles, vocab, max_len)
        if smiles:
            n_covered += 1

    log.info(
        "SMILES matrix: %d/%d drugs have SMILES, vocab_size=%d, max_len=%d",
        n_covered,
        n_drugs,
        len(vocab),
        max_len,
    )
    return matrix, vocab


def get_smiles_matrix(
    drug_to_idx: dict[str, int],
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, dict[str, int]]:
    """Load or build SMILES character matrix for a drug set."""
    smiles_path = processed_dir / "drug_smiles.json"
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"SMILES cache not found: {smiles_path}. Run drug_features.py first."
        )

    with smiles_path.open() as f:
        smiles_dict: dict[str, str | None] = json.load(f)

    return build_smiles_matrix(drug_to_idx, smiles_dict)
