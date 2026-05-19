"""Tests for TrainingConfig dataclass."""

from pathlib import Path

import pytest

from src.training.config import TrainingConfig


class TestTrainingConfig:
    def test_defaults(self) -> None:
        c = TrainingConfig()
        assert c.n_epochs == 50
        assert c.lr == 1e-3
        assert c.batch_size is None
        assert c.warmup_epochs == 0
        assert c.weight_decay == 1e-4

    def test_custom_values(self) -> None:
        c = TrainingConfig(n_epochs=100, lr=0.01, warmup_epochs=5)
        assert c.n_epochs == 100
        assert c.lr == 0.01
        assert c.warmup_epochs == 5

    def test_path_coercion(self) -> None:
        c = TrainingConfig(run_dir="/tmp/run", resume_from="/tmp/ckpt.pt")
        assert isinstance(c.run_dir, Path)
        assert isinstance(c.resume_from, Path)

    def test_invalid_n_epochs(self) -> None:
        with pytest.raises(ValueError, match="n_epochs"):
            TrainingConfig(n_epochs=0)

    def test_invalid_lr(self) -> None:
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=0.0)
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=1.5)

    def test_invalid_checkpoint_every(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_every"):
            TrainingConfig(checkpoint_every=0)
