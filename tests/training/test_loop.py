"""Smoke tests for the refactored training loop (loop.py + trainer*.py wrappers).

Verifies that all four public train_* functions run end-to-end for 1 epoch
and return a history dict with the expected keys and correct length.

Uses a tiny synthetic dataset (4 cells, 8 pairs, 8 omics features) and
1-layer MLPs to keep runtime under 1s on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.dataset import MultiOmicsDataset
from src.training.config import TrainingConfig
from src.training.trainer import train
from src.training.trainer_drug import train_drug, train_drug_fp, train_drug_smiles

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

N_CELLS = 12
N_PAIRS = 40
N_OMICS = 8  # 3 RNA + 5 mutations (matches _make_fake_dataset)
N_DRUGS = 4
FP_DIM = 16  # fingerprint width (small for test)
SMILES_LEN = 6  # padded SMILES length
SMILES_VOCAB = 32  # character vocab size


def _make_fake_dataset(n_cells: int = N_CELLS, n_pairs: int = N_PAIRS) -> MultiOmicsDataset:
    rng = np.random.default_rng(0)
    cell_ids = [f"ACH-{i:04d}" for i in range(n_cells)]
    drug_names = [f"Drug{i}" for i in range(N_DRUGS)]

    dr = pd.DataFrame(
        {
            "depmap_id": rng.choice(cell_ids, n_pairs),
            "drug_name": rng.choice(drug_names, n_pairs),
            "ln_ic50": rng.standard_normal(n_pairs).astype(np.float32),
        }
    )
    rna = pd.DataFrame(
        rng.standard_normal((n_cells, 3)).astype(np.float32),
        index=cell_ids,
        columns=[f"g{i}" for i in range(3)],
    )
    muts = pd.DataFrame(
        rng.standard_normal((n_cells, 5)).astype(np.float32),
        index=cell_ids,
        columns=[f"m{i}" for i in range(5)],
    )
    parquet_map = {
        "drug_response.parquet": dr,
        "rna.parquet": rna,
        "mutations.parquet": muts,
    }

    def _read(path: Path | str, *a: Any, **kw: Any) -> pd.DataFrame:
        return parquet_map[str(path).split("/")[-1]]

    with (
        patch("src.data.dataset.pd.read_parquet", side_effect=_read),
        patch("pathlib.Path.exists", return_value=True),
    ):
        return MultiOmicsDataset(omics_to_use=["rna", "mutations"])


def _tiny_config() -> TrainingConfig:
    return TrainingConfig(
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        device="cpu",
    )


_HISTORY_KEYS = {"train_loss", "val_loss", "val_pearson_r", "epoch_secs"}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainSmokeTest:
    """train() — omics-only baseline."""

    def test_returns_history(self) -> None:
        ds = _make_fake_dataset()
        idx = np.arange(len(ds.targets))
        train_idx, val_idx = idx[: len(idx) // 2], idx[len(idx) // 2 :]

        # Wrap Linear so output is squeezed to 1D (targets are 1D)
        model = nn.Sequential(nn.Linear(N_OMICS, 1), nn.Flatten(0))
        history = train(model, ds, train_idx, val_idx, config=_tiny_config())

        assert set(history.keys()) == _HISTORY_KEYS
        assert len(history["train_loss"]) == 1
        assert len(history["val_pearson_r"]) == 1


class TestTrainDrugSmokeTest:
    """train_drug() — integer drug index variant."""

    def test_returns_history(self) -> None:
        ds = _make_fake_dataset()
        n_drugs = ds.n_drugs
        idx = np.arange(len(ds.targets))
        train_idx, val_idx = idx[: len(idx) // 2], idx[len(idx) // 2 :]

        class _DrugModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.omics = nn.Linear(N_OMICS, 8)
                self.emb = nn.Embedding(n_drugs, 8)
                self.out = nn.Linear(16, 1)

            def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
                return self.out(torch.cat([self.omics(x), self.emb(d)], dim=-1)).squeeze(-1)

        history = train_drug(_DrugModel(), ds, train_idx, val_idx, config=_tiny_config())
        assert set(history.keys()) == _HISTORY_KEYS
        assert len(history["train_loss"]) == 1


class TestTrainDrugFpSmokeTest:
    """train_drug_fp() — fingerprint matrix variant."""

    def test_returns_history(self) -> None:
        ds = _make_fake_dataset()
        n_drugs = ds.n_drugs
        rng = np.random.default_rng(1)
        fp_matrix = rng.standard_normal((n_drugs, FP_DIM)).astype(np.float32)
        idx = np.arange(len(ds.targets))
        train_idx, val_idx = idx[: len(idx) // 2], idx[len(idx) // 2 :]

        class _FpModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.out = nn.Linear(N_OMICS + FP_DIM, 1)

            def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
                return self.out(torch.cat([x, d], dim=-1)).squeeze(-1)

        history = train_drug_fp(
            _FpModel(),
            ds,
            train_idx,
            val_idx,
            fp_matrix,
            n_epochs=1,
            batch_size=4,
            device="cpu",
            warmup_epochs=0,  # skip warmup so SequentialLR branch not needed
        )
        assert set(history.keys()) == _HISTORY_KEYS
        assert len(history["train_loss"]) == 1


class TestTrainDrugSmilesSmokeTest:
    """train_drug_smiles() — SMILES index matrix variant."""

    def test_returns_history(self) -> None:
        ds = _make_fake_dataset()
        n_drugs = ds.n_drugs
        rng = np.random.default_rng(2)
        smiles_matrix = rng.integers(0, SMILES_VOCAB, size=(n_drugs, SMILES_LEN), dtype=np.int32)
        idx = np.arange(len(ds.targets))
        train_idx, val_idx = idx[: len(idx) // 2], idx[len(idx) // 2 :]

        class _SmilesModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(SMILES_VOCAB, 4)
                self.out = nn.Linear(N_OMICS + 4, 1)

            def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
                d_emb = self.emb(d).mean(dim=1)
                return self.out(torch.cat([x, d_emb], dim=-1)).squeeze(-1)

        history = train_drug_smiles(
            _SmilesModel(),
            ds,
            train_idx,
            val_idx,
            smiles_matrix,
            n_epochs=1,
            batch_size=4,
            device="cpu",
        )
        assert set(history.keys()) == _HISTORY_KEYS
        assert len(history["train_loss"]) == 1
