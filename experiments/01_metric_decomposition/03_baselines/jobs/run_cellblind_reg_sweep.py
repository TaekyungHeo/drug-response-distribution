"""Stage 3: Cell-blind regularization sweep.

Cell-blind val r typically peaks early (epoch ~10) then declines — a sign of
overfitting to training cell lines.  This stage asks: is overfitting the
binding constraint for cell-blind r, or is the bottleneck the data itself?

Protocol:
  - RNA-only MLP, fixed architecture [512→256→64→1]
  - Grid: dropout [0.2, 0.3, 0.5] × weight_decay [1e-3, 1e-2, 5e-2]
  - 200 epochs per config; record (best val r, epoch at which it occurs)

Interpretation:
  If no config improves beyond the baseline cell-blind r → regularization is
  not the binding constraint; the bottleneck is the data.
  If a config noticeably improves → original experiment was under-regularized
  and the new r is the corrected cell-blind floor.

Usage:
    uv run python3 experiments/01_metric_decomposition/01_baselines/jobs/run_cellblind_reg_sweep.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.data.omics_utils import build_pair_features, load_omics, z_score_normalize
from src.data.splits import cell_blind_split
from src.evaluation.metrics import evaluate_full

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = (
    REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "results" / "stage3"
)
REPORT_DATA = (
    REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "report" / "data" / "metrics.json"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096 if DEVICE == "cuda" else 2048
MAX_EPOCHS = 200
HIDDEN_DIMS = [512, 256, 64]

DROPOUT_GRID = [0.2, 0.3, 0.5]
WD_GRID = [1e-3, 1e-2, 5e-2]


class CellMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def predict_batched(model: CellMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i : i + BATCH_SIZE]).to(DEVICE)
            parts.append(model(xb).cpu().numpy())
    return np.concatenate(parts)


def global_r_np(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.std() < 1e-9:
        return float("nan")
    return float(pearsonr(preds.astype(np.float64), targets.astype(np.float64))[0])  # type: ignore[arg-type]


def train_and_track(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dropout: float,
    weight_decay: float,
) -> dict:
    """Train for MAX_EPOCHS; return val_r curve, best_val_r, and best_epoch."""
    model = CellMLP(X_train.shape[1], HIDDEN_DIMS, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).to(DEVICE),
        torch.from_numpy(y_train).to(DEVICE),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_r = -np.inf
    best_epoch = 0
    best_state: dict = {}
    curve: list[float] = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            opt.step()
        scheduler.step()

        vr = global_r_np(predict_batched(model, X_val), y_val)
        curve.append(vr)
        if vr > best_val_r:
            best_val_r = vr
            best_epoch = epoch + 1  # 1-indexed
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {
        "model": model,
        "val_curve": curve,
        "best_val_r": float(best_val_r),
        "best_epoch": best_epoch,
    }


def main() -> None:
    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    cell_lines = overlap["depmap_id"].tolist()

    dr = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    dr = pd.DataFrame(dr[dr["depmap_id"].isin(cell_lines)].reset_index(drop=True))

    rna_mat, rna_order = load_omics(["rna"], cell_lines)
    pair_X = build_pair_features(dr, rna_mat, rna_order)

    logger.info("Device: %s  Pairs: %d  RNA features: %d", DEVICE, len(dr), rna_mat.shape[1])

    train_idx, val_idx, test_idx = cell_blind_split(dr, seed=42)
    Xtr, Xvl, Xte = z_score_normalize(pair_X[train_idx], pair_X[val_idx], pair_X[test_idx])
    ytr = dr.iloc[train_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    yvl = dr.iloc[val_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    y_te = dr.iloc[test_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    drugs_te = dr.iloc[test_idx]["drug_name"].to_numpy()
    cells_te = dr.iloc[test_idx]["depmap_id"].to_numpy()

    logger.info("Split: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    t0 = time.perf_counter()
    grid_results = []

    for dropout in DROPOUT_GRID:
        for wd in WD_GRID:
            logger.info("  dropout=%.1f  weight_decay=%g", dropout, wd)
            run = train_and_track(Xtr, ytr, Xvl, yvl, dropout, wd)
            test_preds = predict_batched(run["model"], Xte)
            test_m = dict(evaluate_full(y_te, test_preds, drugs_te, cells_te))
            entry = {
                "dropout": dropout,
                "weight_decay": wd,
                "best_val_r": run["best_val_r"],
                "best_epoch": run["best_epoch"],
                "val_curve": [round(v, 5) for v in run["val_curve"]],
                **{k: v for k, v in test_m.items() if k != "n"},
                "n": test_m["n"],
            }
            grid_results.append(entry)
            logger.info(
                "    best_val_r=%.4f @ epoch %d  test_global_r=%.4f"
                "  per_drug_r=%.4f  per_cell_r=%.4f",
                entry["best_val_r"], entry["best_epoch"],
                entry["global_r"], entry["per_drug_r"], entry["per_cell_r"],
            )

    logger.info("Total runtime: %.1f min", (time.perf_counter() - t0) / 60)

    best = max(grid_results, key=lambda x: x["best_val_r"])
    logger.info(
        "Best config: dropout=%.1f wd=%g  best_val_r=%.4f @ epoch %d  test_global_r=%.4f",
        best["dropout"], best["weight_decay"], best["best_val_r"],
        best["best_epoch"], best["global_r"],
    )

    results = {
        "grid": grid_results,
        "best": {k: v for k, v in best.items() if k != "val_curve"},
        "architecture": HIDDEN_DIMS,
        "max_epochs": MAX_EPOCHS,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "cellblind_reg_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("Results: %s", out)

    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REPORT_DATA.read_text()) if REPORT_DATA.exists() else {}
    existing["cellblind_reg_sweep"] = results
    REPORT_DATA.write_text(json.dumps(existing, indent=2))
    logger.info("Report data: %s", REPORT_DATA)


if __name__ == "__main__":
    main()
