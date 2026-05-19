"""TransformerEncoder training loop for cross-split and model-comparison jobs.

Extracted from jobs/run.py so it can be reused without code duplication.
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

N_EPOCHS = 50
WARMUP_EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 512
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
MODALITY_DROPOUT_P = 0.3
OMICS = ["rna", "mutations"]


class _Prefetcher:
    def __init__(self, concat_np, cell_rows, drug_idxs, fp_matrix,
                 targets, indices, bs, device):
        self._c, self._cr, self._di, self._fp = concat_np, cell_rows, drug_idxs, fp_matrix
        self._t, self._idx, self._bs, self._dev = targets, indices, bs, device
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
                    idx = self._idx[perm[i:i + self._bs]]
                    rows = self._cr[idx]
                    x = torch.from_numpy(self._c[rows].copy()).to(self._dev)
                    fp = torch.from_numpy(self._fp[self._di[idx]].copy()).to(self._dev)
                    y = torch.from_numpy(self._t[idx].copy()).to(self._dev)
                    self._q.put((x, fp, y), timeout=60)
        except Exception as e:
            self._err = e

    def __next__(self):
        if self._err:
            raise RuntimeError(str(self._err))
        return self._q.get(timeout=60)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def train_transformer_fold(
    concat_np: np.ndarray,
    cell_rows: np.ndarray,
    drug_idxs: np.ndarray,
    fp_matrix: np.ndarray,
    targets: np.ndarray,
    drug_names_all: np.ndarray,
    cell_ids_all: np.ndarray,
    feature_dims: dict[str, int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    fold_label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train TransformerEncoder for one fold.

    Returns (test_preds, test_targets, test_drug_names, test_cell_ids).
    """
    from src.models.transformer_encoder import TransformerEncoder

    model = TransformerEncoder(
        feature_dims=feature_dims,
        modality_order=OMICS,
        drug_fp_dim=fp_matrix.shape[1],
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, modality_dropout_p=MODALITY_DROPOUT_P,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, N_EPOCHS - WARMUP_EPOCHS), eta_min=LR * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    criterion = nn.MSELoss()

    steps = max(1, len(train_idx) // BATCH_SIZE)
    best_val_r = -np.inf
    best_state: dict | None = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        pf = _Prefetcher(concat_np, cell_rows, drug_idxs, fp_matrix,
                         targets, train_idx, BATCH_SIZE, device)
        for _ in range(steps):
            x, fp, y = next(pf)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, fp)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pf.stop()
        scheduler.step()

        model.eval()
        preds_v: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(val_idx), BATCH_SIZE * 2):
                chunk = val_idx[i:i + BATCH_SIZE * 2]
                rows = cell_rows[chunk]
                x = torch.from_numpy(concat_np[rows].copy()).to(device)
                fp_b = torch.from_numpy(fp_matrix[drug_idxs[chunk]].copy()).to(device)
                preds_v.append(model(x, fp_b).cpu().numpy())
        val_preds = np.concatenate(preds_v)
        val_r = float(pearsonr(targets[val_idx], val_preds)[0])

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info("[%s TransformerEncoder] ep %3d/%d val_r=%.4f",
                        fold_label, epoch, N_EPOCHS, val_r)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    preds_t: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(test_idx), BATCH_SIZE * 2):
            chunk = test_idx[i:i + BATCH_SIZE * 2]
            rows = cell_rows[chunk]
            x = torch.from_numpy(concat_np[rows].copy()).to(device)
            fp_b = torch.from_numpy(fp_matrix[drug_idxs[chunk]].copy()).to(device)
            preds_t.append(model(x, fp_b).cpu().numpy())
    test_preds = np.concatenate(preds_t)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return (
        test_preds,
        targets[test_idx],
        drug_names_all[test_idx],
        cell_ids_all[test_idx],
    )
