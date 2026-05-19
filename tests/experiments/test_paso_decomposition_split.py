"""Tests for the drug-blind val split used in 02_reproductions/01_paso/02_decomposition.

The split function is the only logic in run.py that is unit-testable without a model
or real data. The key invariant it must satisfy:

  val drugs ∩ train drugs = ∅

The previous random-pair split violated this — val pairs were drawn from training drugs,
inflating val_r to ~0.91 while test_r (drug-blind) stayed ~0.34. That made val_r an
unreliable signal for stopping: a model that memorised training drugs looked great on val
but bad on test. The drug-blind split puts val and test on the same distribution so the
snooping-inflation measurement is valid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

# Import the split function directly from the experiment script.
# The script has no import-time side effects beyond standard library + numpy.
sys.path.insert(0, str(ROOT / "experiments" / "02_reproductions" / "01_paso" / "02_decomposition" / "jobs"))
from run import split_drug_blind_val  # noqa: E402


def _make_inputs(
    n_pairs: int = 200,
    n_drugs: int = 20,
    n_train: int = 160,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic drug_idxs_arr and full_train_idx."""
    rng = np.random.default_rng(seed)
    drug_idxs_arr = rng.integers(0, n_drugs, size=n_pairs).astype(np.int32)
    full_train_idx = np.arange(n_train, dtype=np.int64)
    return drug_idxs_arr, full_train_idx


class TestSplitDrugBlindVal:
    def test_val_drugs_disjoint_from_train_drugs(self) -> None:
        """Core invariant: no val drug appears in training."""
        drug_idxs_arr, full_train_idx = _make_inputs()
        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=0)

        train_drugs = set(drug_idxs_arr[train_idx].tolist())  # type: ignore[index]
        val_drugs = set(drug_idxs_arr[val_idx].tolist())
        assert train_drugs & val_drugs == set(), (
            f"Overlap: {train_drugs & val_drugs}"
        )

    def test_val_drugs_subset_of_original_train_drugs(self) -> None:
        """Val drugs come from PASO's training set, not test set."""
        drug_idxs_arr, full_train_idx = _make_inputs()
        _train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=0)

        original_train_drugs = set(drug_idxs_arr[full_train_idx].tolist())
        val_drugs = set(drug_idxs_arr[val_idx].tolist())
        assert val_drugs <= original_train_drugs

    def test_train_and_val_cover_all_full_train(self) -> None:
        """Every index in full_train_idx ends up in exactly one of train or val."""
        drug_idxs_arr, full_train_idx = _make_inputs()
        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=0)

        combined = np.sort(np.concatenate([train_idx, val_idx]))
        assert np.array_equal(combined, np.sort(full_train_idx))

    def test_val_drug_count_is_ten_percent(self) -> None:
        """Number of held-out drugs = max(1, floor(n_train_drugs * 0.10))."""
        drug_idxs_arr, full_train_idx = _make_inputs(n_drugs=20)
        _train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=0)

        n_original_drugs = len(np.unique(drug_idxs_arr[full_train_idx]))
        n_val_drugs = len(np.unique(drug_idxs_arr[val_idx]))
        expected = max(1, int(n_original_drugs * 0.10))
        assert n_val_drugs == expected, f"n_val_drugs={n_val_drugs}, expected={expected}"

    def test_reproducible_with_same_fold_i(self) -> None:
        """Same fold_i always produces the same split."""
        drug_idxs_arr, full_train_idx = _make_inputs()
        train_a, val_a = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=3)
        train_b, val_b = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=3)
        assert np.array_equal(np.sort(train_a), np.sort(train_b))
        assert np.array_equal(np.sort(val_a), np.sort(val_b))

    def test_different_fold_i_gives_different_val_drugs(self) -> None:
        """Different folds should (almost always) select different val drug sets."""
        drug_idxs_arr, full_train_idx = _make_inputs(n_drugs=30)
        _, val_0 = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=0)
        _, val_1 = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=1)
        drugs_0 = set(drug_idxs_arr[val_0].tolist())
        drugs_1 = set(drug_idxs_arr[val_1].tolist())
        assert drugs_0 != drugs_1

    def test_val_is_non_empty(self) -> None:
        drug_idxs_arr, full_train_idx = _make_inputs()
        _train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=0)
        assert len(val_idx) > 0

    def test_train_larger_than_val(self) -> None:
        drug_idxs_arr, full_train_idx = _make_inputs()
        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=0)
        assert len(train_idx) > len(val_idx)

    def test_single_drug_edge_case(self) -> None:
        """With only one drug in training, it must go to val (max(1, 0) = 1)."""
        drug_idxs_arr = np.zeros(100, dtype=np.int32)
        full_train_idx = np.arange(50, dtype=np.int64)
        train_idx, val_idx = split_drug_blind_val(drug_idxs_arr, full_train_idx, fold_i=0)
        assert len(np.unique(drug_idxs_arr[val_idx])) == 1
        # With one drug → all pairs go to val, train is empty
        assert len(train_idx) == 0

    def test_custom_val_frac(self) -> None:
        """val_frac=0.20 should hold out ~20% of drugs."""
        drug_idxs_arr, full_train_idx = _make_inputs(n_drugs=20)
        _train_idx, val_idx = split_drug_blind_val(
            drug_idxs_arr, full_train_idx, fold_i=0, val_frac=0.20
        )
        n_original_drugs = len(np.unique(drug_idxs_arr[full_train_idx]))
        n_val_drugs = len(np.unique(drug_idxs_arr[val_idx]))
        expected = max(1, int(n_original_drugs * 0.20))
        assert n_val_drugs == expected
