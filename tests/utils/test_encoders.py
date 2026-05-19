"""Tests for Strategy pattern encoder interfaces (src/utils/encoders.py).

Covers: DrugEncoder/CellEncoder Protocol compliance, EncoderRegistry,
and built-in null encoders.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.encoders import (
    CellEncoder,
    DrugEncoder,
    EncoderRegistry,
    random_drug_encoder,
    zero_drug_encoder,
)

# ── Protocol compliance (structural subtyping) ─────────────────────────────


class TestDrugEncoderProtocol:
    """Any object with encode(list[str]) -> np.ndarray satisfies DrugEncoder."""

    def test_zero_encoder_is_callable(self) -> None:
        # zero_drug_encoder is a DrugEncoderFn (Callable), not a DrugEncoder Protocol
        # (Protocol requires an .encode() method; functions are called directly)
        assert callable(zero_drug_encoder)
        result = zero_drug_encoder(["A", "B"])
        assert result.shape == (2, 2048)

    def test_lambda_is_callable(self) -> None:
        fn = lambda ids: np.zeros((len(ids), 64), dtype=np.float32)  # noqa: E731
        assert callable(fn)
        result = fn(["X"])
        assert result.shape == (1, 64)

    def test_class_satisfies_protocol(self) -> None:
        class MyEncoder:
            def encode(self, ids: list[str]) -> np.ndarray:
                return np.zeros((len(ids), 128), dtype=np.float32)

        enc = MyEncoder()
        assert isinstance(enc, DrugEncoder)

    def test_class_without_encode_fails(self) -> None:
        class BadEncoder:
            def transform(self, ids: list[str]) -> np.ndarray:
                return np.zeros((len(ids), 64), dtype=np.float32)

        bad = BadEncoder()
        assert not isinstance(bad, DrugEncoder)


class TestCellEncoderProtocol:
    def test_class_satisfies_protocol(self) -> None:
        class RNAEncoder:
            def encode(self, cell_ids: list[str]) -> np.ndarray:
                return np.random.rand(len(cell_ids), 512).astype(np.float32)

        enc = RNAEncoder()
        assert isinstance(enc, CellEncoder)


# ── Built-in null encoders ─────────────────────────────────────────────────


class TestZeroDrugEncoder:
    def test_shape(self) -> None:
        result = zero_drug_encoder(["SMILES1", "SMILES2", "SMILES3"])
        assert result.shape == (3, 2048)

    def test_custom_dim(self) -> None:
        result = zero_drug_encoder(["A"], dim=64)
        assert result.shape == (1, 64)

    def test_dtype_float32(self) -> None:
        result = zero_drug_encoder(["A", "B"])
        assert result.dtype == np.float32

    def test_all_zeros(self) -> None:
        result = zero_drug_encoder(["A", "B", "C"])
        assert np.all(result == 0.0)

    def test_empty_list(self) -> None:
        result = zero_drug_encoder([])
        assert result.shape == (0, 2048)


class TestRandomDrugEncoder:
    def test_shape(self) -> None:
        result = random_drug_encoder(["A", "B"])
        assert result.shape == (2, 2048)

    def test_dtype_float32(self) -> None:
        result = random_drug_encoder(["A"])
        assert result.dtype == np.float32

    def test_deterministic_with_seed(self) -> None:
        r1 = random_drug_encoder(["A", "B"], seed=42)
        r2 = random_drug_encoder(["A", "B"], seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self) -> None:
        r1 = random_drug_encoder(["A"], seed=0)
        r2 = random_drug_encoder(["A"], seed=1)
        assert not np.array_equal(r1, r2)


# ── EncoderRegistry ───────────────────────────────────────────────────────


class TestEncoderRegistry:
    def setup_method(self) -> None:
        # Snapshot registry state; restore after each test
        self._orig = dict(EncoderRegistry._registry)

    def teardown_method(self) -> None:
        EncoderRegistry._registry.clear()
        EncoderRegistry._registry.update(self._orig)

    def test_builtin_zeros_registered(self) -> None:
        assert "zeros" in EncoderRegistry.available()

    def test_builtin_random_registered(self) -> None:
        assert "random" in EncoderRegistry.available()

    def test_get_builtin_zeros(self) -> None:
        enc = EncoderRegistry.get("zeros")
        result = enc(["A", "B"])
        assert result.shape == (2, 2048)
        assert np.all(result == 0.0)

    def test_register_custom(self) -> None:
        def tiny(ids: list[str]) -> np.ndarray:
            return np.ones((len(ids), 4), dtype=np.float32)

        EncoderRegistry.register("tiny", tiny)
        assert "tiny" in EncoderRegistry.available()
        result = EncoderRegistry.get("tiny")(["X"])
        assert result.shape == (1, 4)

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            EncoderRegistry.get("nonexistent_encoder")

    def test_available_returns_sorted_list(self) -> None:
        names = EncoderRegistry.available()
        assert names == sorted(names)
