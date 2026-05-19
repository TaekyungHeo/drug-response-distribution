"""Optimized training loop for M5 Pro: MPS compute + CPU prefetch threading.

from __future__ import annotations

Measured bottlenecks (M5 Pro, 64 GB, ConcatBaseline, bs=2048):
  - numpy fancy index (CPU): ~10 ms/batch
  - .to('mps') transfer:     ~8 ms/batch
  - MPS fwd+bwd:             ~67 ms/batch
  Strategy: prefetch thread prepares batch i+1 on CPU while MPS runs batch i.
  Theoretical limit: 2048 / 0.067s ≈ 30K samp/s vs naive 16K.

Safe batch size for MPS: 2048 (bs=4096 triggers MPS internal assertion bug).

Checkpointing:
  - Saved every checkpoint_every epochs to run_dir/checkpoints/
  - Resume by passing resume_from=<checkpoint_path>
  - Checkpoint contains: model state, optimizer state, scheduler state, epoch, history
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import MultiOmicsDataset
from src.training.config import TrainingConfig

logger = logging.getLogger(__name__)

DEVICE = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

DEFAULT_BATCH_SIZE = {"mps": 2048, "cuda": 4096, "cpu": 2048}


# ---------------------------------------------------------------------------
# Prefetch thread
# ---------------------------------------------------------------------------


class _Prefetcher:
    """Background thread: numpy index → torch tensor → device transfer.

    Overlaps CPU data prep with GPU compute. Queue depth=2 ensures GPU never
    starves while still bounding memory usage to ~2 × batch_size × features.
    """

    def __init__(
        self,
        concat_np: np.ndarray,
        cell_rows: np.ndarray,
        targets: np.ndarray,
        pair_indices: np.ndarray,
        batch_size: int,
        device: str,
        queue_depth: int = 2,
    ) -> None:
        self._concat = concat_np
        self._cell_rows = cell_rows
        self._targets = targets
        self._pair_indices = pair_indices
        self._bs = batch_size
        self._device = device
        self._q: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._stop = threading.Event()
        self._err: Exception | None = None
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        try:
            n = len(self._pair_indices)
            perm = np.random.permutation(n)
            i = 0
            while not self._stop.is_set():
                if i + self._bs > n:
                    perm = np.random.permutation(n)
                    i = 0
                batch_pair_idx = perm[i : i + self._bs]
                i += self._bs
                rows = self._cell_rows[self._pair_indices[batch_pair_idx]]
                x_np = self._concat[rows].copy()
                y_np = self._targets[self._pair_indices[batch_pair_idx]].copy()
                x = torch.from_numpy(x_np).to(self._device)
                y = torch.from_numpy(y_np).to(self._device)
                self._q.put((x, y), timeout=60)
        except Exception as e:
            self._err = e

    def __next__(self) -> None:
        if self._err is not None:
            raise RuntimeError(f"Prefetch thread failed: {self._err}") from self._err
        return self._q.get(timeout=60)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_concat(dataset: MultiOmicsDataset) -> np.ndarray:
    """Thin wrapper around dataset.to_concat_array() for backward compatibility."""
    return dataset.to_concat_array()


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    history: dict[str, list],
    best_val_r: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_val_r": best_val_r,
        },
        path,
    )
    logger.debug("checkpoint saved → %s", path.name)


# ---------------------------------------------------------------------------
# Shared forward / val helpers for omics-only model
# ---------------------------------------------------------------------------


def _forward_omics_only(mdl: nn.Module, inputs: tuple) -> torch.Tensor:
    (x,) = inputs
    return mdl(x)


def _val_step_omics(
    mdl: nn.Module,
    concat_np: np.ndarray,
    ds: MultiOmicsDataset,
    vidx: np.ndarray,
    dev: str,
    bs: int,
) -> tuple[list, list, list]:
    criterion = nn.MSELoss()
    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    loss_list: list[float] = []
    for i in range(0, len(vidx), bs * 2):
        chunk = vidx[i : i + bs * 2]
        rows = ds.cell_rows[chunk]
        x = torch.from_numpy(concat_np[rows].copy()).to(dev)
        y_np = ds.targets[chunk]
        y = torch.from_numpy(y_np.copy()).to(dev)
        p = mdl(x)
        loss_list.append(criterion(p, y).item())
        preds_list.append(p.cpu().numpy())
        targets_list.append(y_np)
    return preds_list, targets_list, loss_list


# ---------------------------------------------------------------------------
# Training loop (thin wrapper → _run_training_loop)
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    dataset: MultiOmicsDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: TrainingConfig | None = None,
    # Legacy keyword args (keep for backward compat)
    n_epochs: int = 50,
    batch_size: int | None = None,
    lr: float = 1e-3,
    device: str | None = None,
    run_dir: Path | None = None,
    checkpoint_every: int = 10,
    resume_from: Path | None = None,
    model_name: str = "model",
) -> dict[str, list]:
    # Merge legacy kwargs into a TrainingConfig
    if config is None:
        config = TrainingConfig(
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            run_dir=run_dir,
            checkpoint_every=checkpoint_every,
            resume_from=resume_from,
            model_name=model_name,
        )

    from src.training.loop import _run_training_loop

    def _prefetcher_factory(
        concat_np: np.ndarray, ds: MultiOmicsDataset, idx: np.ndarray, bs: int, dev: str
    ) -> _Prefetcher:
        return _Prefetcher(concat_np, ds.cell_rows, ds.targets, idx, bs, dev)

    return _run_training_loop(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        config=config,
        prefetcher_factory=_prefetcher_factory,
        forward_step=_forward_omics_only,
        val_step=_val_step_omics,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(
    model: nn.Module,
    dataset: MultiOmicsDataset,
    idx: np.ndarray,
    device: str | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    if device is None:
        device = DEVICE
    model = model.to(device)
    model.eval()
    concat_np = _build_concat(dataset)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            chunk = idx[i : i + batch_size]
            rows = dataset.cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(device)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds)
