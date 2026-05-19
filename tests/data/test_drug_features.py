"""Tests for drug feature utilities (annotation stripping, fingerprint computation)."""

import numpy as np
import pytest

from src.data.drug_features import _strip_annotations, compute_fingerprints


def _fingerprints_available() -> bool:
    try:
        from rdkit import Chem  # noqa: F401

        return True
    except ImportError:
        return False


def test_strip_annotations_removes_concentration() -> None:
    assert _strip_annotations("Bleomycin (50 uM)") == "Bleomycin"


def test_strip_annotations_removes_stereo() -> None:
    assert _strip_annotations("Nutlin-3a (-)") == "Nutlin-3a"


def test_strip_annotations_trailing_whitespace() -> None:
    assert _strip_annotations("GSK-LSD1-2HCl ") == "GSK-LSD1-2HCl"


def test_strip_annotations_no_change() -> None:
    assert _strip_annotations("Imatinib") == "Imatinib"


def test_strip_annotations_keeps_internal_parens() -> None:
    # Parenthetical mid-name, not at string end — unchanged
    result = _strip_annotations("KRAS (G12C) Inhibitor-12")
    assert result == "KRAS (G12C) Inhibitor-12"


@pytest.mark.skipif(not _fingerprints_available(), reason="rdkit not available")
def test_compute_fingerprints_shape() -> None:
    smiles_dict = {
        "Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "Missing": None,
    }
    fps = compute_fingerprints(smiles_dict, {"Imatinib": 0, "Missing": 1}, n_bits=512)
    assert fps.shape == (2, 512)
    assert fps.dtype == np.float32


@pytest.mark.skipif(not _fingerprints_available(), reason="rdkit not available")
def test_compute_fingerprints_missing_is_zeros() -> None:
    fps = compute_fingerprints({"Drug": None}, {"Drug": 0}, n_bits=512)
    assert (fps[0] == 0).all()


@pytest.mark.skipif(not _fingerprints_available(), reason="rdkit not available")
def test_compute_fingerprints_binary_values() -> None:
    fps = compute_fingerprints({"Aspirin": "CC(=O)Oc1ccccc1C(=O)O"}, {"Aspirin": 0}, n_bits=2048)
    assert set(fps[0].tolist()) <= {0.0, 1.0}
