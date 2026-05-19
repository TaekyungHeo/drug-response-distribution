"""Drug-aware training loop: extends trainer.py to pass drug indices alongside omics.

from __future__ import annotations

Measured bottlenecks and device constraints are identical to trainer.py.
Critical: non_blocking=True is FORBIDDEN on MPS — causes GC race with async DMA.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import MultiOmicsDataset
from src.training.config import TrainingConfig
from src.training.trainer import (
    DEVICE,
    _build_concat,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drug-aware prefetch thread (unified: handles both index and matrix drug repr)
# ---------------------------------------------------------------------------


class _DrugPrefetcher:
    """Background thread: numpy index → (x_omics, drug_repr, y) tensors → device.

    When drug_matrix is None, yields integer drug indices directly
    (for embedding-based models like TransformerEncoderGNN).  When drug_matrix is
    provided, looks up drug_idxs → drug_matrix rows and yields those tensors
    (for fingerprint/SMILES-based models).

    Queue depth=2 keeps device busy without excess memory overhead.
    """

    def __init__(
        self,
        concat_np: np.ndarray,
        cell_rows: np.ndarray,
        drug_idxs: np.ndarray,
        targets: np.ndarray,
        pair_indices: np.ndarray,
        batch_size: int,
        device: str,
        drug_matrix: np.ndarray | None = None,
        queue_depth: int = 2,
    ) -> None:
        self._concat = concat_np
        self._cell_rows = cell_rows
        self._drug_idxs = drug_idxs
        self._targets = targets
        self._pair_indices = pair_indices
        self._bs = batch_size
        self._device = device
        self._drug_matrix = drug_matrix
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
                pair_drug_idxs = self._drug_idxs[self._pair_indices[batch_pair_idx]]
                if self._drug_matrix is not None:
                    drug_np = self._drug_matrix[pair_drug_idxs].copy()
                else:
                    drug_np = pair_drug_idxs.copy()
                y_np = self._targets[self._pair_indices[batch_pair_idx]].copy()
                x = torch.from_numpy(x_np).to(self._device)
                d = torch.from_numpy(drug_np).to(self._device)
                y = torch.from_numpy(y_np).to(self._device)
                self._q.put((x, d, y), timeout=60)
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
# Shared forward / val helpers (reused across wrappers)
# ---------------------------------------------------------------------------


def _forward_x_d(mdl: nn.Module, inputs: tuple) -> torch.Tensor:
    x, d = inputs
    return mdl(x, d)


def _forward_x_d_long(mdl: nn.Module, inputs: tuple) -> torch.Tensor:
    x, d = inputs
    return mdl(x, d.long())


def _val_step_drug_idx(
    mdl: nn.Module,
    concat_np: np.ndarray,
    ds: MultiOmicsDataset,
    vidx: np.ndarray,
    dev: str,
    bs: int,
) -> tuple[list, list, list]:
    """Val loop for models taking integer drug indices (train_drug)."""
    criterion = torch.nn.MSELoss()
    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    loss_list: list[float] = []
    for i in range(0, len(vidx), bs * 2):
        chunk = vidx[i : i + bs * 2]
        rows = ds.cell_rows[chunk]
        x = torch.from_numpy(concat_np[rows].copy()).to(dev)
        d = torch.from_numpy(ds.drug_indices[chunk].copy()).to(dev)
        y_np = ds.targets[chunk]
        y = torch.from_numpy(y_np.copy()).to(dev)
        p = mdl(x, d)
        loss_list.append(criterion(p, y).item())
        preds_list.append(p.cpu().numpy())
        targets_list.append(y_np)
    return preds_list, targets_list, loss_list


def _make_matrix_val_step(
    drug_matrix: np.ndarray,
    *,
    use_long: bool = False,
    bs_multiplier: int = 2,
) -> Callable:
    """Return a val_step closure for matrix-based drug models.

    Args:
        drug_matrix: Drug representation matrix (fp or SMILES indices).
        use_long: Cast drug tensor to LongTensor (required for SMILES/embedding).
        bs_multiplier: Chunk size = bs * bs_multiplier. Use 2 (fp) or 1 (smiles).
    """

    def _val_step(
        mdl: nn.Module,
        concat_np: np.ndarray,
        ds: MultiOmicsDataset,
        vidx: np.ndarray,
        dev: str,
        bs: int,
    ) -> tuple[list, list, list]:
        criterion = torch.nn.MSELoss()
        preds_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        loss_list: list[float] = []
        chunk_size = bs * bs_multiplier
        for i in range(0, len(vidx), chunk_size):
            chunk = vidx[i : i + chunk_size]
            rows = ds.cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(dev)
            d = torch.from_numpy(drug_matrix[ds.drug_indices[chunk]].copy()).to(dev)
            if use_long:
                d = d.long()
            y_np = ds.targets[chunk]
            y = torch.from_numpy(y_np.copy()).to(dev)
            p = mdl(x, d)
            if not use_long:
                loss_list.append(criterion(p, y).item())
            preds_list.append(p.cpu().numpy())
            targets_list.append(y_np)
        # For SMILES (use_long=True), val_loss computed post-hoc in loop.py
        return preds_list, targets_list, loss_list

    return _val_step


# ---------------------------------------------------------------------------
# Training loops (thin wrappers → _run_training_loop)
# ---------------------------------------------------------------------------


def train_drug(
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
    warmup_epochs: int = 0,
) -> dict[str, list]:
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
            warmup_epochs=warmup_epochs,
        )

    from src.training.loop import _run_training_loop

    def _prefetcher_factory(
        concat_np: np.ndarray, ds: MultiOmicsDataset, idx: np.ndarray, bs: int, dev: str
    ) -> _DrugPrefetcher:
        return _DrugPrefetcher(concat_np, ds.cell_rows, ds.drug_indices, ds.targets, idx, bs, dev)

    return _run_training_loop(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        config=config,
        prefetcher_factory=_prefetcher_factory,
        forward_step=_forward_x_d,
        val_step=_val_step_drug_idx,
    )


def train_drug_fp(
    model: nn.Module,
    dataset: MultiOmicsDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    drug_fp_matrix: np.ndarray,
    n_epochs: int = 100,
    batch_size: int | None = None,
    lr: float = 1e-3,
    device: str | None = None,
    run_dir: Path | None = None,
    checkpoint_every: int = 10,
    resume_from: Path | None = None,
    model_name: str = "model",
    warmup_epochs: int = 10,
    config: TrainingConfig | None = None,
) -> dict[str, list]:
    """Train a fingerprint-based model (TransformerEncoder or DrugFingerprintMLP).

    Passes x_drug_fp: (batch, 2048) float32 fingerprint tensors. Uses LinearLR
    warmup for warmup_epochs then CosineAnnealingLR for the remainder.
    """
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
            warmup_epochs=warmup_epochs,
        )

    from src.training.loop import _run_training_loop

    _fp = drug_fp_matrix

    def _prefetcher_factory(
        concat_np: np.ndarray, ds: MultiOmicsDataset, idx: np.ndarray, bs: int, dev: str
    ) -> _DrugPrefetcher:
        return _DrugPrefetcher(
            concat_np, ds.cell_rows, ds.drug_indices, ds.targets, idx, bs, dev, drug_matrix=_fp
        )

    return _run_training_loop(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        config=config,
        prefetcher_factory=_prefetcher_factory,
        forward_step=_forward_x_d,
        val_step=_make_matrix_val_step(_fp),
    )


def train_drug_smiles(
    model: nn.Module,
    dataset: MultiOmicsDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    drug_smiles_matrix: np.ndarray,
    n_epochs: int = 100,
    batch_size: int | None = None,
    lr: float = 1e-3,
    device: str | None = None,
    run_dir: Path | None = None,
    checkpoint_every: int = 25,
    model_name: str = "smiles_model",
    warmup_epochs: int = 0,
    config: TrainingConfig | None = None,
) -> dict[str, list]:
    """Training loop for TransformerEncoderSmiles (SMILES character sequences as drug input).

    Identical to train_drug_fp but passes int32 SMILES index sequences instead of
    float32 fingerprint vectors. The model receives LongTensor (batch, max_len).
    """
    if config is None:
        config = TrainingConfig(
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            run_dir=run_dir,
            checkpoint_every=checkpoint_every,
            resume_from=None,  # smiles variant historically has no resume_from param
            model_name=model_name,
            warmup_epochs=warmup_epochs,
        )

    from src.training.loop import _run_training_loop

    _sm = drug_smiles_matrix

    # Use _DrugPrefetcher with drug_matrix — it works for any 2D drug matrix (int or float)
    def _prefetcher_factory(
        concat_np: np.ndarray, ds: MultiOmicsDataset, idx: np.ndarray, bs: int, dev: str
    ) -> _DrugPrefetcher:
        return _DrugPrefetcher(
            concat_np, ds.cell_rows, ds.drug_indices, ds.targets, idx, bs, dev, drug_matrix=_sm
        )

    return _run_training_loop(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        config=config,
        prefetcher_factory=_prefetcher_factory,
        forward_step=_forward_x_d_long,
        val_step=_make_matrix_val_step(_sm, use_long=True, bs_multiplier=1),
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _predict_with_matrix(
    model: nn.Module,
    dataset: MultiOmicsDataset,
    idx: np.ndarray,
    drug_matrix: np.ndarray,
    device: str,
    batch_size: int,
    *,
    use_long: bool = False,
) -> np.ndarray:
    """Shared inference loop for matrix-based drug representations."""
    concat_np = _build_concat(dataset)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            chunk = idx[i : i + batch_size]
            rows = dataset.cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(device)
            d = torch.from_numpy(drug_matrix[dataset.drug_indices[chunk]].copy()).to(device)
            if use_long:
                d = d.long()
            preds.append(model(x, d).cpu().numpy())
    return np.concatenate(preds)


def predict_drug(
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
            d = torch.from_numpy(dataset.drug_indices[chunk].copy()).to(device)
            preds.append(model(x, d).cpu().numpy())
    return np.concatenate(preds)


def predict_drug_fp(
    model: nn.Module,
    dataset: MultiOmicsDataset,
    idx: np.ndarray,
    drug_fp_matrix: np.ndarray,
    device: str | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Run inference for a fingerprint-based model (TransformerEncoder or DrugFingerprintMLP)."""
    if device is None:
        device = DEVICE
    model = model.to(device)
    model.eval()
    return _predict_with_matrix(model, dataset, idx, drug_fp_matrix, device, batch_size)


def predict_drug_smiles(
    model: nn.Module,
    dataset: MultiOmicsDataset,
    idx: np.ndarray,
    drug_smiles_matrix: np.ndarray,
    device: str | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Inference for TransformerEncoderSmiles."""
    if device is None:
        device = DEVICE
    model = model.to(device)
    model.eval()
    return _predict_with_matrix(
        model, dataset, idx, drug_smiles_matrix, device, batch_size, use_long=True
    )
