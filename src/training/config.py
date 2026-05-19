"""Training configuration dataclass shared across all trainer variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["TrainingConfig"]


@dataclass
class TrainingConfig:
    n_epochs: int = 50
    lr: float = 1e-3
    batch_size: int | None = None  # None → auto-detected per device
    device: str | None = None  # None → DEVICE auto-detect
    run_dir: Path | None = None
    checkpoint_every: int = 10
    resume_from: Path | None = None
    model_name: str = "model"
    warmup_epochs: int = 0  # was only in train_drug; unify here
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        # Validate
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be > 0, got {self.n_epochs}")
        if not 0 < self.lr <= 1.0:
            raise ValueError(f"lr must be in (0, 1], got {self.lr}")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be > 0")
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)
