"""Tests for data split strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import cell_blind_split, drug_blind_split, mixed_set_split


@pytest.fixture
def dummy_pairs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cell_lines = [f"ACH-{i:06d}" for i in range(100)]
    drugs = [f"Drug_{i}" for i in range(20)]
    rows = [(c, d, rng.normal()) for c in cell_lines for d in drugs[:5]]
    return pd.DataFrame(rows, columns=["depmap_id", "drug_name", "ln_ic50"])


def test_mixed_set_split_sizes(dummy_pairs: pd.DataFrame) -> None:
    train, val, test = mixed_set_split(dummy_pairs, val_frac=0.1, test_frac=0.2)
    n = len(dummy_pairs)
    assert len(train) + len(val) + len(test) == n
    assert len(test) == pytest.approx(n * 0.2, abs=2)


def test_mixed_set_no_overlap(dummy_pairs: pd.DataFrame) -> None:
    train, val, test = mixed_set_split(dummy_pairs)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0


def test_cell_blind_no_cell_line_leakage(dummy_pairs: pd.DataFrame) -> None:
    train, val, test = cell_blind_split(dummy_pairs)
    train_lines = set(dummy_pairs.iloc[train]["depmap_id"])
    test_lines = set(dummy_pairs.iloc[test]["depmap_id"])
    val_lines = set(dummy_pairs.iloc[val]["depmap_id"])
    assert len(train_lines & test_lines) == 0
    assert len(train_lines & val_lines) == 0


def test_drug_blind_no_drug_leakage(dummy_pairs: pd.DataFrame) -> None:
    train, val, test = drug_blind_split(dummy_pairs)
    train_drugs = set(dummy_pairs.iloc[train]["drug_name"])
    test_drugs = set(dummy_pairs.iloc[test]["drug_name"])
    val_drugs = set(dummy_pairs.iloc[val]["drug_name"])
    assert len(train_drugs & test_drugs) == 0
    assert len(train_drugs & val_drugs) == 0
