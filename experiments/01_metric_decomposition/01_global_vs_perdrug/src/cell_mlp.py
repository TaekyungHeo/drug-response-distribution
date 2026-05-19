"""CellDrugMLP: simple concat-MLP for (PCA omics + Morgan FP) → IC50.

Used in Step 3 (model comparison) to compare capacity-controlled MLP variants
against Ridge and TransformerEncoder on the same feature set.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# Architectures: (hidden_sizes, label)
SIZES: dict[str, list[int]] = {
    "S": [128, 64],
    "M": [512, 256, 64],
    "L": [2048, 512, 128],
}

MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
LR_MIN = 1e-5
BATCH_SIZE = 512
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
WARMUP_EPOCHS = 5


class CellDrugMLP(nn.Module):
    """BN → (Linear → BN → ReLU → Dropout) × n → Linear(1)."""

    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float = DROPOUT) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _Prefetcher:
    """Background thread that pre-fetches minibatches to GPU."""

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                 bs: int, device: str) -> None:
        self._X, self._y, self._idx = X, y, indices
        self._bs, self._dev = bs, device
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._err: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            n = len(self._idx)
            rng = np.random.default_rng()
            while not self._stop.is_set():
                perm = rng.permutation(n)
                for i in range(0, n - self._bs + 1, self._bs):
                    if self._stop.is_set():
                        break
                    chunk = self._idx[perm[i:i + self._bs]]
                    x = torch.from_numpy(self._X[chunk]).to(self._dev)
                    y = torch.from_numpy(self._y[chunk]).to(self._dev)
                    self._q.put((x, y), timeout=60)
        except Exception as e:
            self._err = e

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._err:
            raise RuntimeError(str(self._err))
        return self._q.get(timeout=60)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def train_mlp_fold(
    X_all: np.ndarray,
    y_all: np.ndarray,
    drug_names_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    size: str,
    device: str,
    fold_label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train one MLP fold and return (test_preds, test_targets, test_drugs).

    X_all: (n_pairs, input_dim) — already concatenated [pca_omics | morgan_fp]
    with z-score normalisation applied using training-set statistics.
    """
    hidden = SIZES[size]
    input_dim = X_all.shape[1]

    # Z-score normalise using training set stats
    mean = X_all[train_idx].mean(axis=0)
    std = X_all[train_idx].std(axis=0)
    std[std < 1e-9] = 1.0
    X_norm = ((X_all - mean) / std).astype(np.float32)

    model = CellDrugMLP(input_dim, hidden, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, MAX_EPOCHS - WARMUP_EPOCHS), eta_min=LR_MIN)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    criterion = nn.MSELoss()

    steps_per_epoch = max(1, len(train_idx) // BATCH_SIZE)
    best_val_r = -np.inf
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        pf = _Prefetcher(X_norm, y_all, train_idx, BATCH_SIZE, device)
        for _ in range(steps_per_epoch):
            x, y = next(pf)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pf.stop()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            chunks = [X_norm[val_idx[i:i + BATCH_SIZE * 4]]
                      for i in range(0, len(val_idx), BATCH_SIZE * 4)]
            val_preds = np.concatenate([
                model(torch.from_numpy(c).to(device)).cpu().numpy() for c in chunks
            ])
        val_r = float(pearsonr(y_all[val_idx], val_preds)[0])

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info("[%s MLP-%s] ep %3d val_r=%.4f best=%.4f",
                        fold_label, size, epoch, val_r, best_val_r)

        if no_improve >= PATIENCE:
            logger.info("[%s MLP-%s] early stop at epoch %d", fold_label, size, epoch)
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        chunks = [X_norm[test_idx[i:i + BATCH_SIZE * 4]]
                  for i in range(0, len(test_idx), BATCH_SIZE * 4)]
        test_preds = np.concatenate([
            model(torch.from_numpy(c).to(device)).cpu().numpy() for c in chunks
        ])

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return test_preds, y_all[test_idx], drug_names_all[test_idx]
