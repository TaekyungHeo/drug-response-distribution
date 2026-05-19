"""Unit tests for src/utils/paso_folds.py — uses tmp_path for fake CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.paso_folds import (
    build_pair_index,
    load_paso_folds,
    load_paso_pairs,
    map_fold_indices,
    split_drug_blind_val,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FOLD_COLS = ["drug", "cell_line", "IC50"]


def _write_fold_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_fold_dir(tmp_path: Path, n_folds: int = 2) -> Path:
    """Create a fake PASO fold directory with n_folds train/test CSVs."""
    for i in range(n_folds):
        _write_fold_csv(
            tmp_path / f"DrugBlind_train_Fold{i}.csv",
            [
                {"drug": "aspirin", "cell_line": "A549", "IC50": 1.0},
                {"drug": "ibuprofen", "cell_line": "MCF7", "IC50": 2.0},
            ],
        )
        _write_fold_csv(
            tmp_path / f"DrugBlind_test_Fold{i}.csv",
            [
                {"drug": "aspirin", "cell_line": "MCF7", "IC50": 1.5},
            ],
        )
    return tmp_path


# ---------------------------------------------------------------------------
# load_paso_folds
# ---------------------------------------------------------------------------


class TestLoadPasoFolds:
    def test_returns_correct_number_of_folds(self, tmp_path: Path) -> None:
        d = _make_fold_dir(tmp_path, n_folds=3)
        folds = load_paso_folds(n_folds=3, paso_dir=d)
        assert len(folds) == 3

    def test_fold_contents(self, tmp_path: Path) -> None:
        d = _make_fold_dir(tmp_path, n_folds=1)
        folds = load_paso_folds(n_folds=1, paso_dir=d)
        train_df, test_df = folds[0]
        assert list(train_df.columns) == FOLD_COLS
        assert len(train_df) == 2
        assert len(test_df) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_paso_folds(n_folds=1, paso_dir=tmp_path)


# ---------------------------------------------------------------------------
# build_pair_index
# ---------------------------------------------------------------------------


class TestBuildPairIndex:
    @pytest.fixture()
    def setup(self, tmp_path: Path) -> tuple:
        d = _make_fold_dir(tmp_path, n_folds=1)
        folds = load_paso_folds(n_folds=1, paso_dir=d)
        name_to_depmap = {"A549": "ACH-001", "MCF7": "ACH-002"}
        rna_index = pd.Index(["ACH-001", "ACH-002"])
        mut_index = pd.Index(["ACH-001", "ACH-002"])
        return folds, name_to_depmap, rna_index, mut_index

    def test_full_df_columns(self, setup: tuple) -> None:
        folds, n2d, rna, mut = setup
        full_df, _ = build_pair_index(folds, n2d, rna, mut)
        assert set(full_df.columns) == {"depmap_id", "drug_name", "ic50"}

    def test_key_to_idx_mapping(self, setup: tuple) -> None:
        folds, n2d, rna, mut = setup
        full_df, key_to_idx = build_pair_index(folds, n2d, rna, mut)
        for (dep, drug), idx in key_to_idx.items():
            assert full_df.loc[idx, "depmap_id"] == dep
            assert full_df.loc[idx, "drug_name"] == drug

    def test_filters_missing_omics(self, tmp_path: Path) -> None:
        d = _make_fold_dir(tmp_path, n_folds=1)
        folds = load_paso_folds(n_folds=1, paso_dir=d)
        name_to_depmap = {"A549": "ACH-001", "MCF7": "ACH-002"}
        # MCF7 missing from mut_index -> should be filtered out
        rna_index = pd.Index(["ACH-001", "ACH-002"])
        mut_index = pd.Index(["ACH-001"])
        full_df, _ = build_pair_index(folds, name_to_depmap, rna_index, mut_index)
        assert "ACH-002" not in full_df["depmap_id"].values


# ---------------------------------------------------------------------------
# map_fold_indices
# ---------------------------------------------------------------------------


class TestMapFoldIndices:
    def test_correct_indices(self, tmp_path: Path) -> None:
        d = _make_fold_dir(tmp_path, n_folds=1)
        folds = load_paso_folds(n_folds=1, paso_dir=d)
        name_to_depmap = {"A549": "ACH-001", "MCF7": "ACH-002"}
        rna_index = pd.Index(["ACH-001", "ACH-002"])
        mut_index = pd.Index(["ACH-001", "ACH-002"])
        full_df, key_to_idx = build_pair_index(folds, name_to_depmap, rna_index, mut_index)
        indices = map_fold_indices(folds[0][0], key_to_idx, name_to_depmap)
        assert indices.dtype == np.int64
        assert len(indices) > 0
        # All indices should be valid rows in full_df
        assert all(i in full_df.index for i in indices)

    def test_unknown_cell_skipped(self) -> None:
        df = pd.DataFrame([{"drug": "x", "cell_line": "UNKNOWN", "IC50": 1.0}])
        key_to_idx: dict = {}
        name_to_depmap: dict = {}
        indices = map_fold_indices(df, key_to_idx, name_to_depmap)
        assert len(indices) == 0


# ---------------------------------------------------------------------------
# split_drug_blind_val
# ---------------------------------------------------------------------------


class TestSplitDrugBlindVal:
    def _make_pairs(self, n_drugs: int = 20, n_pairs_per_drug: int = 10) -> np.ndarray:
        """Return drug_idxs_arr where each drug appears n_pairs_per_drug times."""
        return np.repeat(np.arange(n_drugs, dtype=np.int32), n_pairs_per_drug)

    def test_disjoint(self) -> None:
        drug_idxs = self._make_pairs(n_drugs=20)
        full_train_idx = np.arange(len(drug_idxs), dtype=np.int64)
        train_idx, val_idx = split_drug_blind_val(drug_idxs, full_train_idx, fold_i=0)
        assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_covers_full_train(self) -> None:
        drug_idxs = self._make_pairs(n_drugs=20)
        full_train_idx = np.arange(len(drug_idxs), dtype=np.int64)
        train_idx, val_idx = split_drug_blind_val(drug_idxs, full_train_idx, fold_i=0)
        assert len(train_idx) + len(val_idx) == len(full_train_idx)

    def test_val_fraction(self) -> None:
        drug_idxs = self._make_pairs(n_drugs=20)
        full_train_idx = np.arange(len(drug_idxs), dtype=np.int64)
        _, val_idx = split_drug_blind_val(drug_idxs, full_train_idx, fold_i=0, val_frac=0.10)
        # 10% of 20 drugs = 2 val drugs → val_idx should have ~20 pairs
        val_drugs = np.unique(drug_idxs[val_idx])
        n_drugs_total = len(np.unique(drug_idxs))
        assert 1 <= len(val_drugs) <= max(1, int(n_drugs_total * 0.15))

    def test_drug_blind(self) -> None:
        drug_idxs = self._make_pairs(n_drugs=20)
        full_train_idx = np.arange(len(drug_idxs), dtype=np.int64)
        train_idx, val_idx = split_drug_blind_val(drug_idxs, full_train_idx, fold_i=0)
        train_drugs = set(drug_idxs[train_idx].tolist())
        val_drugs = set(drug_idxs[val_idx].tolist())
        assert train_drugs.isdisjoint(val_drugs)

    def test_deterministic(self) -> None:
        drug_idxs = self._make_pairs(n_drugs=20)
        full_train_idx = np.arange(len(drug_idxs), dtype=np.int64)
        t1, v1 = split_drug_blind_val(drug_idxs, full_train_idx, fold_i=3)
        t2, v2 = split_drug_blind_val(drug_idxs, full_train_idx, fold_i=3)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)

    def test_fold_seed_varies(self) -> None:
        drug_idxs = self._make_pairs(n_drugs=20)
        full_train_idx = np.arange(len(drug_idxs), dtype=np.int64)
        _, v0 = split_drug_blind_val(drug_idxs, full_train_idx, fold_i=0)
        _, v1 = split_drug_blind_val(drug_idxs, full_train_idx, fold_i=1)
        val_drugs_0 = set(drug_idxs[v0].tolist())
        val_drugs_1 = set(drug_idxs[v1].tolist())
        # Different folds should (typically) produce different val drug sets
        assert val_drugs_0 != val_drugs_1


# ---------------------------------------------------------------------------
# load_paso_pairs
# ---------------------------------------------------------------------------


class TestLoadPasoPairs:
    def _make_fold_csv(self, tmp_path: Path) -> Path:
        train_rows = [
            {"drug": "aspirin", "cell_line": "A549", "IC50": 1.0},
            {"drug": "ibuprofen", "cell_line": "A549", "IC50": 2.0},
            {"drug": "aspirin", "cell_line": "MCF7", "IC50": 1.5},
        ]
        test_rows = [
            {"drug": "paracetamol", "cell_line": "MCF7", "IC50": 0.5},
        ]
        pd.DataFrame(train_rows).to_csv(tmp_path / "DrugBlind_train_Fold0.csv", index=False)
        pd.DataFrame(test_rows).to_csv(tmp_path / "DrugBlind_test_Fold0.csv", index=False)
        return tmp_path

    def test_returns_train_test(self, tmp_path: Path) -> None:
        d = self._make_fold_csv(tmp_path)
        name_to_depmap = {"A549": "ACH-001", "MCF7": "ACH-002"}
        available = {"ACH-001", "ACH-002"}
        train_df, test_df = load_paso_pairs(d, name_to_depmap, available, k=0)
        assert set(train_df.columns) == {"depmap_id", "drug_name", "ln_ic50"}
        assert set(test_df.columns) == {"depmap_id", "drug_name", "ln_ic50"}

    def test_filters_unavailable_cells(self, tmp_path: Path) -> None:
        d = self._make_fold_csv(tmp_path)
        name_to_depmap = {"A549": "ACH-001", "MCF7": "ACH-002"}
        available = {"ACH-001"}  # MCF7/ACH-002 not available
        train_df, test_df = load_paso_pairs(d, name_to_depmap, available, k=0)
        assert all(c == "ACH-001" for c in train_df["depmap_id"])
        assert len(test_df) == 0  # paracetamol/MCF7 filtered out

    def test_depmap_mapping(self, tmp_path: Path) -> None:
        d = self._make_fold_csv(tmp_path)
        name_to_depmap = {"A549": "ACH-001", "MCF7": "ACH-002"}
        available = {"ACH-001", "ACH-002"}
        train_df, _ = load_paso_pairs(d, name_to_depmap, available, k=0)
        assert set(train_df["depmap_id"]).issubset({"ACH-001", "ACH-002"})
