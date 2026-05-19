"""Encoder protocols and registry — Strategy pattern via Python Protocols.

GoF Strategy pattern in Python 3.9+ is best expressed as Protocol-based
interfaces rather than abstract base classes. This allows structural subtyping:
any object with the right methods is accepted, without explicit inheritance.

Usage:
    from src.utils.encoders import DrugEncoder, CellEncoder, EncoderRegistry

    # Any callable matching the protocol is a valid encoder
    class MorganEncoder:
        def encode(self, ids: list[str]) -> np.ndarray: ...

    # Or use the registry to look up by name
    encoder = EncoderRegistry.get("morgan")
    features = encoder.encode(drug_smiles)
"""

from __future__ import annotations

from typing import Callable, ClassVar, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DrugEncoder(Protocol):
    """Strategy interface for drug feature encoding.

    Implementations: Morgan fingerprints, ChemBERTa, GNN, LINCS signatures.
    Any object with an ``encode`` method matching this signature is valid.
    """

    def encode(self, smiles_or_ids: list[str]) -> np.ndarray:
        """Encode drugs to a fixed-length feature matrix.

        Args:
            smiles_or_ids: list of SMILES strings or drug identifiers
        Returns:
            shape (n_drugs, feature_dim), dtype float32
        """
        ...


@runtime_checkable
class CellEncoder(Protocol):
    """Strategy interface for cell feature encoding.

    Implementations: RNA PCA, scFoundation embeddings, pathway scores.
    """

    def encode(self, cell_ids: list[str]) -> np.ndarray:
        """Encode cell lines to a fixed-length feature matrix.

        Args:
            cell_ids: list of DepMap IDs
        Returns:
            shape (n_cells, feature_dim), dtype float32
        """
        ...


# ---------------------------------------------------------------------------
# Lightweight function-based encoders (Strategy as Callable)
# ---------------------------------------------------------------------------

# Type aliases for function-based strategy
DrugEncoderFn = Callable[[list[str]], np.ndarray]
CellEncoderFn = Callable[[list[str]], np.ndarray]


class EncoderRegistry:
    """Registry pattern for named encoder lookup.

    Allows experiment scripts to select encoders by string name
    rather than importing specific classes directly.

    Example:
        EncoderRegistry.register("zeros", lambda ids: np.zeros((len(ids), 2048)))
        encoder = EncoderRegistry.get("zeros")
    """

    _registry: ClassVar[dict[str, DrugEncoderFn]] = {}

    @classmethod
    def register(cls, name: str, encoder_fn: DrugEncoderFn) -> None:
        """Register a named encoder function."""
        cls._registry[name] = encoder_fn

    @classmethod
    def get(cls, name: str) -> DrugEncoderFn:
        """Retrieve a registered encoder by name."""
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            msg = f"Encoder '{name}' not registered. Available: {available}"
            raise KeyError(msg)
        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        """List all registered encoder names."""
        return sorted(cls._registry.keys())


# ---------------------------------------------------------------------------
# Built-in null encoders (for ablation experiments)
# ---------------------------------------------------------------------------


def zero_drug_encoder(smiles_or_ids: list[str], dim: int = 2048) -> np.ndarray:
    """Return all-zeros drug features (null encoder for ablation)."""
    return np.zeros((len(smiles_or_ids), dim), dtype=np.float32)


def random_drug_encoder(
    smiles_or_ids: list[str],
    dim: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    """Return random drug features (random baseline for ablation)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(smiles_or_ids), dim)).astype(np.float32)


# Register built-ins
EncoderRegistry.register("zeros", zero_drug_encoder)
EncoderRegistry.register("random", random_drug_encoder)


__all__ = [
    "CellEncoder",
    "CellEncoderFn",
    "DrugEncoder",
    "DrugEncoderFn",
    "EncoderRegistry",
    "random_drug_encoder",
    "zero_drug_encoder",
]
