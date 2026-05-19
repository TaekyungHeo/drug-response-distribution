"""Training utilities for the 03_baselines experiment.

Primary val metric: per-drug r (macro-averaged Pearson r within each drug, ≥5 samples).
Early stopping: patience=30 epochs on val per-drug r.
Convergence check: warns if best checkpoint epoch ≥ max_epochs - patience.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn

# Imported lazily to allow sys.path to be set before this module is used.
# Callers must have repo root on sys.path before importing train.


def _mean_per_drug_r(preds: np.ndarray, targets: np.ndarray, drug_names: np.ndarray) -> float:
    from src.evaluation.per_drug import mean_per_drug_r  # repo-level src
    return mean_per_drug_r(preds.astype(np.float64), targets.astype(np.float64), drug_names)


def predict_batched(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            parts.append(model(xb).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_drugs: np.ndarray,
    hidden_dims: list[int],
    dropout: float,
    weight_decay: float,
    device: str,
    *,
    max_epochs: int = 300,
    patience: int = 30,
    batch_size: int = 4096,
) -> tuple[nn.Module, list[float], float, int]:
    """Train a CellMLP and return the best checkpoint by val per-drug r.

    Returns:
        (best_model, val_per_drug_r_curve, best_val_per_drug_r, best_epoch_1indexed)

    A UserWarning is emitted if best_epoch >= max_epochs - patience, indicating the
    run may not have converged (PLAN threshold: epoch ≥ 270 for max_epochs=300, patience=30).
    """
    from src.models import CellMLP  # type: ignore[import]  # resolved via sys.path at runtime

    model = CellMLP(X_train.shape[1], hidden_dims, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val_r = -np.inf
    best_epoch = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    curve: list[float] = []
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            opt.step()
        scheduler.step()

        val_preds = predict_batched(model, X_val, device, batch_size)
        vr = _mean_per_drug_r(val_preds, y_val, val_drugs)
        curve.append(float(vr) if not np.isnan(vr) else float("nan"))

        if vr > best_val_r:
            best_val_r = float(vr)
            best_epoch = epoch + 1  # 1-indexed
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)

    convergence_threshold = max_epochs - patience
    if max_epochs > patience and best_epoch >= convergence_threshold:
        warnings.warn(
            f"Best checkpoint at epoch {best_epoch} >= {convergence_threshold} "
            f"(max_epochs={max_epochs} - patience={patience}). "
            "Run may be unconverged; re-run with more epochs if used for conclusions.",
            UserWarning,
            stacklevel=2,
        )

    return model, curve, best_val_r, best_epoch
