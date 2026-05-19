"""Core training loop extracted from trainer.py / trainer_drug.py.

Template Method via function injection: the ~67% identical training algorithm
lives here. Only the prefetcher factory and forward/validation steps vary per
variant.

Critical device notes (inherited from trainer.py):
  - non_blocking=True is FORBIDDEN on MPS — causes GC race with async DMA.
  - Safe batch size for MPS: 2048 (bs=4096 triggers MPS internal assertion bug).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import MultiOmicsDataset
from src.evaluation.metrics import evaluate
from src.training.config import TrainingConfig
from src.training.trainer import DEFAULT_BATCH_SIZE, DEVICE, _save_checkpoint, _sync

logger = logging.getLogger(__name__)


@runtime_checkable
class BatchIterator(Protocol):
    """Protocol: anything that yields batches and has a .stop() method."""

    def __next__(self) -> tuple: ...

    def stop(self) -> None: ...


# Type aliases
PrefetcherFactory = Callable[..., BatchIterator]
ForwardStep = Callable[[nn.Module, tuple], torch.Tensor]
ValStep = Callable[
    [nn.Module, np.ndarray, MultiOmicsDataset, np.ndarray, str, int],
    tuple[list[np.ndarray], list[np.ndarray], list[float]],
]


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    lr: float,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build cosine scheduler, optionally with linear warm-up."""
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=lr * 0.01
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)


def _run_training_loop(
    model: nn.Module,
    dataset: MultiOmicsDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: TrainingConfig,
    prefetcher_factory: PrefetcherFactory,
    forward_step: ForwardStep,
    val_step: ValStep,
) -> dict[str, list]:
    """Single training loop implementation shared by all trainer variants.

    Args:
        model: PyTorch model to train.
        dataset: MultiOmicsDataset with pairs.
        train_idx: Indices into dataset for training pairs.
        val_idx: Indices into dataset for validation pairs.
        config: TrainingConfig with all hyperparameters.
        prefetcher_factory: Callable(concat_np, dataset, train_idx, batch_size, device)
                            → BatchIterator for training.
        forward_step: Callable(model, batch) → predictions tensor.
                      The batch tuple is everything except the last element (y).
        val_step: Callable(model, concat_np, dataset, val_idx, device, batch_size)
                  → (preds_list, targets_list, val_loss_list).

    Returns:
        History dict with train_loss, val_loss, val_pearson_r, epoch_secs.
    """
    # Resolve config fields to local variables
    n_epochs = config.n_epochs
    lr = config.lr
    batch_size = config.batch_size
    device = config.device
    run_dir = config.run_dir
    checkpoint_every = config.checkpoint_every
    resume_from = config.resume_from
    model_name = config.model_name
    warmup_epochs = config.warmup_epochs

    if device is None:
        device = DEVICE
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE.get(device, 2048)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(optimizer, n_epochs, lr, warmup_epochs)
    criterion = nn.MSELoss()

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_pearson_r": [],
        "epoch_secs": [],
    }
    best_val_r = -np.inf
    best_state = None
    start_epoch = 1

    # Resume from checkpoint if provided
    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt["history"]
        best_val_r = ckpt["best_val_r"]
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from checkpoint %s (epoch %d)", resume_from, ckpt["epoch"])

    concat_np = dataset.to_concat_array()
    steps_per_epoch = len(train_idx) // batch_size
    samples_per_epoch = steps_per_epoch * batch_size

    logger.info(
        "Training %s | device=%s  bs=%d  steps/epoch=%d  epochs=%d→%d  warmup=%d",
        model_name,
        device,
        batch_size,
        steps_per_epoch,
        start_epoch,
        n_epochs,
        warmup_epochs,
    )

    total_t0 = time.perf_counter()

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()

        prefetcher = prefetcher_factory(concat_np, dataset, train_idx, batch_size, device)
        train_losses: list[float] = []
        for _step in range(steps_per_epoch):
            batch = next(prefetcher)
            # Last element is always y; everything before is model input
            *inputs, y = batch
            optimizer.zero_grad(set_to_none=True)
            preds = forward_step(model, tuple(inputs))
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        prefetcher.stop()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds_list, val_targets_list, val_loss_list = val_step(
                model, concat_np, dataset, val_idx, device, batch_size
            )

        _sync(device)
        epoch_secs = time.perf_counter() - epoch_t0
        samp_per_sec = samples_per_epoch / epoch_secs

        val_preds_arr = np.concatenate(val_preds_list)
        val_targets_arr = np.concatenate(val_targets_list)
        metrics = evaluate(val_targets_arr, val_preds_arr)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(
            float(np.mean(val_loss_list))
            if val_loss_list
            else float(
                criterion(torch.from_numpy(val_preds_arr), torch.from_numpy(val_targets_arr)).item()
            )
        )
        history["val_pearson_r"].append(metrics["pearson_r"])
        history["epoch_secs"].append(epoch_secs)

        if metrics["pearson_r"] > best_val_r:
            best_val_r = metrics["pearson_r"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed_total = time.perf_counter() - total_t0
        epochs_left = n_epochs - epoch
        eta_secs = (elapsed_total / (epoch - start_epoch + 1)) * epochs_left

        logger.info(
            "[%s] epoch %3d/%d  train=%.4f  val=%.4f  r=%.4f  %.0f samp/s  %.1fs/epoch  ETA %.0fm",
            model_name,
            epoch,
            n_epochs,
            history["train_loss"][-1],
            history["val_loss"][-1],
            metrics["pearson_r"],
            samp_per_sec,
            epoch_secs,
            eta_secs / 60,
        )

        # Checkpoint
        if run_dir is not None and epoch % checkpoint_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"{model_name}_epoch{epoch:03d}.pt"
            _save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, history, best_val_r)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save final checkpoint
    if run_dir is not None:
        ckpt_path = run_dir / "checkpoints" / f"{model_name}_final.pt"
        _save_checkpoint(ckpt_path, n_epochs, model, optimizer, scheduler, history, best_val_r)

    return history
