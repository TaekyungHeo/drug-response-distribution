"""Stage 2: Fixed-capacity MLP — capacity sweep and modality ablation.

Sub-experiment A — Capacity sweep (RNA-only):
  Architectures: Small [128→64→1] / Medium [512→256→64→1] / Large [2048→512→128→1]
  Hyperparameters: dropout [0.1, 0.3, 0.5] × weight_decay [0, 1e-4, 1e-3]
  Best config per capacity level selected on mixed-set val set.
  Each winning config evaluated on all three splits.

Sub-experiment B — Modality ablation (Medium capacity, best config from A):
  RNA-only → +mutations → +CNV → +metabolomics → all-5-omics
  Evaluated on all three splits.

Answers:
  Does capacity limit the result?    (Large ≈ Medium → no)
  Does more omics help?              (all-5 > RNA-only at fixed capacity → yes)

Usage:
    uv run python3 experiments/01_metric_decomposition/01_baselines/jobs/run_mlp_sweep.py
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

REPO_ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.data.omics_utils import build_pair_features, load_omics, z_score_normalize
from src.data.splits import cell_blind_split, drug_blind_split, mixed_set_split
from src.evaluation.metrics import evaluate_full

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = (
    REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "results" / "stage2"
)
REPORT_DATA = (
    REPO_ROOT / "experiments" / "01_metric_decomposition" / "01_baselines" / "report" / "data" / "metrics.json"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096 if DEVICE == "cuda" else 2048
MAX_EPOCHS = 200

CAPACITY_GRID: dict[str, list[int]] = {
    "small":  [128, 64],
    "medium": [512, 256, 64],
    "large":  [2048, 512, 128],
}

DROPOUT_GRID = [0.1, 0.3, 0.5]
WD_GRID = [0.0, 1e-4, 1e-3]

SPLIT_FNS = {
    "mixed_set": mixed_set_split,
    "cell_blind": cell_blind_split,
    "drug_blind": drug_blind_split,
}

MODALITY_VARIANTS: list[tuple[str, list[str]]] = [
    ("rna_only",        ["rna"]),
    ("rna_mut",         ["rna", "mutations"]),
    ("rna_mut_cnv",     ["rna", "mutations", "cnv"]),
    ("rna_mut_cnv_met", ["rna", "mutations", "cnv", "metabolomics"]),
    ("all_5_omics",     ["rna", "mutations", "cnv", "metabolomics", "rppa"]),
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def predict_batched(model: CellMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i : i + BATCH_SIZE]).to(DEVICE)
            parts.append(model(xb).cpu().numpy())
    return np.concatenate(parts)


def global_r_np(preds: np.ndarray, targets: np.ndarray) -> float:
    from scipy.stats import pearsonr
    if preds.std() < 1e-9:
        return float("nan")
    return float(pearsonr(preds.astype(np.float64), targets.astype(np.float64))[0])  # type: ignore[arg-type]


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int],
    dropout: float,
    weight_decay: float,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[CellMLP, list[float], float]:
    """Train; return (best_model_by_val_r, val_r_curve, best_val_r)."""
    model = CellMLP(X_train.shape[1], hidden_dims, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).to(DEVICE),
        torch.from_numpy(y_train).to(DEVICE),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_r = -np.inf
    best_state: dict = {}
    curve: list[float] = []

    for _ in range(max_epochs):
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, curve, best_val_r


# ---------------------------------------------------------------------------
# Sub-experiment A: capacity sweep
# ---------------------------------------------------------------------------

def run_capacity_sweep(dr: pd.DataFrame, pair_X: np.ndarray) -> dict:
    logger.info("=== Sub-experiment A: capacity sweep (RNA-only, mixed-set) ===")
    train_idx, val_idx, _ = mixed_set_split(dr, seed=42)
    Xtr0, Xvl0, _ = z_score_normalize(pair_X[train_idx], pair_X[val_idx], pair_X[val_idx])
    ytr = dr.iloc[train_idx]["ln_ic50"].to_numpy(dtype=np.float32)
    yvl = dr.iloc[val_idx]["ln_ic50"].to_numpy(dtype=np.float32)

    best_configs: dict[str, dict] = {}
    for cap_name, hidden_dims in CAPACITY_GRID.items():
        best_vr, best_do, best_wd = -np.inf, DROPOUT_GRID[0], WD_GRID[0]
        for do in DROPOUT_GRID:
            for wd in WD_GRID:
                _, _, vr = train_mlp(Xtr0, ytr, Xvl0, yvl, hidden_dims, do, wd)
                if vr > best_vr:
                    best_vr, best_do, best_wd = vr, do, wd
        best_configs[cap_name] = {"dropout": best_do, "weight_decay": best_wd, "val_r": float(best_vr)}
        logger.info("  [%s] best dropout=%.1f wd=%g val_r=%.4f", cap_name, best_do, best_wd, best_vr)

    # Evaluate best config per capacity on all splits
    eval_results: dict[str, dict] = {}
    for cap_name, hidden_dims in CAPACITY_GRID.items():
        cfg = best_configs[cap_name]
        eval_results[cap_name] = {"config": cfg, "splits": {}}
        for split_name, split_fn in SPLIT_FNS.items():
            tr_i, vl_i, te_i = split_fn(dr, seed=42)
            Xtr, Xvl, Xte = z_score_normalize(pair_X[tr_i], pair_X[vl_i], pair_X[te_i])
            ytr2 = dr.iloc[tr_i]["ln_ic50"].to_numpy(dtype=np.float32)
            yvl2 = dr.iloc[vl_i]["ln_ic50"].to_numpy(dtype=np.float32)
            model, curve, _ = train_mlp(Xtr, ytr2, Xvl, yvl2, hidden_dims, cfg["dropout"], cfg["weight_decay"])
            preds = predict_batched(model, Xte)
            y_te = dr.iloc[te_i]["ln_ic50"].to_numpy(dtype=np.float32)
            m = dict(evaluate_full(y_te, preds, dr.iloc[te_i]["drug_name"].to_numpy(), dr.iloc[te_i]["depmap_id"].to_numpy()))
            m["val_curve"] = [round(v, 5) for v in curve]
            eval_results[cap_name]["splits"][split_name] = m
            logger.info("  [%s][%s]  global_r=%.4f  per_drug_r=%.4f  per_cell_r=%.4f",
                        cap_name, split_name, m["global_r"], m["per_drug_r"], m["per_cell_r"])

    return {"capacity_sweep": eval_results, "best_configs": best_configs}


# ---------------------------------------------------------------------------
# Sub-experiment B: modality ablation
# ---------------------------------------------------------------------------

def run_modality_ablation(
    dr: pd.DataFrame,
    cell_lines: list[str],
    best_medium_cfg: dict,
) -> dict:
    logger.info("\n=== Sub-experiment B: modality ablation (medium capacity) ===")
    hidden_dims = CAPACITY_GRID["medium"]
    dropout = float(best_medium_cfg["dropout"])
    weight_decay = float(best_medium_cfg["weight_decay"])

    results: dict = {}
    for variant_name, modalities in MODALITY_VARIANTS:
        cell_mat, cell_order = load_omics(modalities, cell_lines)
        pair_X = build_pair_features(dr, cell_mat, cell_order)
        results[variant_name] = {"modalities": modalities, "n_features": int(pair_X.shape[1]), "splits": {}}

        for split_name, split_fn in SPLIT_FNS.items():
            tr_i, vl_i, te_i = split_fn(dr, seed=42)
            Xtr, Xvl, Xte = z_score_normalize(pair_X[tr_i], pair_X[vl_i], pair_X[te_i])
            ytr = dr.iloc[tr_i]["ln_ic50"].to_numpy(dtype=np.float32)
            yvl = dr.iloc[vl_i]["ln_ic50"].to_numpy(dtype=np.float32)
            model, curve, _ = train_mlp(Xtr, ytr, Xvl, yvl, hidden_dims, dropout, weight_decay)
            preds = predict_batched(model, Xte)
            y_te = dr.iloc[te_i]["ln_ic50"].to_numpy(dtype=np.float32)
            m = dict(evaluate_full(y_te, preds, dr.iloc[te_i]["drug_name"].to_numpy(), dr.iloc[te_i]["depmap_id"].to_numpy()))
            m["val_curve"] = [round(v, 5) for v in curve]
            results[variant_name]["splits"][split_name] = m
            logger.info("  [%s][%s]  global_r=%.4f  per_drug_r=%.4f  per_cell_r=%.4f",
                        variant_name, split_name, m["global_r"], m["per_drug_r"], m["per_cell_r"])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    overlap = pd.read_parquet(PROCESSED_DIR / "overlap_cell_lines.parquet")
    cell_lines = overlap["depmap_id"].tolist()

    dr = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    dr = pd.DataFrame(dr[dr["depmap_id"].isin(cell_lines)].reset_index(drop=True))
    logger.info("Device: %s  Pairs: %d  Cells: %d  Drugs: %d",
                DEVICE, len(dr), dr["depmap_id"].nunique(), dr["drug_name"].nunique())

    rna_mat, rna_order = load_omics(["rna"], cell_lines)
    rna_pair_X = build_pair_features(dr, rna_mat, rna_order)
    logger.info("RNA features: %d", rna_mat.shape[1])

    t0 = time.perf_counter()

    sweep = run_capacity_sweep(dr, rna_pair_X)
    ablation = run_modality_ablation(dr, cell_lines, sweep["best_configs"]["medium"])

    logger.info("Total runtime: %.1f min", (time.perf_counter() - t0) / 60)

    final = {
        "capacity_sweep": sweep["capacity_sweep"],
        "best_configs": sweep["best_configs"],
        "modality_ablation": ablation,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "mlp_sweep_results.json"
    out.write_text(json.dumps(final, indent=2))
    logger.info("Results: %s", out)

    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REPORT_DATA.read_text()) if REPORT_DATA.exists() else {}
    existing["mlp_sweep"] = final
    REPORT_DATA.write_text(json.dumps(existing, indent=2))
    logger.info("Report data: %s", REPORT_DATA)


if __name__ == "__main__":
    main()
