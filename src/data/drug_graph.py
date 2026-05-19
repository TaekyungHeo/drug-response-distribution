"""Convert drug SMILES to padded dense graph representations for GCN training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

log = logging.getLogger(__name__)

# Standard atom types used across drug-response GNN literature (43 explicit + "other" = 44)
_ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
]  # 43 entries

# Hybridization types (6 explicit + "other" = 7)
if RDKIT_AVAILABLE:
    _HYBRIDIZATIONS = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
        rdchem.HybridizationType.S,
    ]
else:
    _HYBRIDIZATIONS = []

# 44 + 11 + 5 + 5 + 1 + 1 + 7 = 74
D_ATOM: int = 74
MAX_ATOMS: int = 100


def _atom_features(atom: rdchem.Atom) -> np.ndarray:
    """74-dim feature vector for a single RDKit atom.

    Breakdown:
        [0:44]   atom type one-hot (43 explicit + "other")
        [44:55]  degree one-hot (0-9 + "≥10")
        [55:60]  formal charge one-hot (clamped to [-2, 2])
        [60:65]  num Hs one-hot (clamped to [0, 4])
        [65]     is aromatic
        [66]     is in ring
        [67:74]  hybridization one-hot (6 explicit + "other")
    """
    feats: list[float] = []

    # Atom type (44)
    symbol = atom.GetSymbol()
    feats += [float(symbol == t) for t in _ATOM_TYPES]
    feats.append(float(symbol not in _ATOM_TYPES))

    # Degree (11): 0-9 + "≥10"
    deg = atom.GetDegree()
    feats += [float(deg == i) for i in range(10)]
    feats.append(float(deg >= 10))

    # Formal charge (5): clamp to [-2, 2]
    charge = max(-2, min(2, atom.GetFormalCharge()))
    feats += [float(charge == c) for c in [-2, -1, 0, 1, 2]]

    # Num Hs (5): clamp to [0, 4]
    nhs = min(4, atom.GetTotalNumHs())
    feats += [float(nhs == i) for i in range(5)]

    # Aromaticity (1)
    feats.append(float(atom.GetIsAromatic()))

    # In ring (1)
    feats.append(float(atom.IsInRing()))

    # Hybridization (7): 6 explicit + "other"
    hyb = atom.GetHybridization()
    feats += [float(hyb == h) for h in _HYBRIDIZATIONS]
    feats.append(float(hyb not in _HYBRIDIZATIONS))

    assert len(feats) == D_ATOM, f"Expected {D_ATOM}, got {len(feats)}"
    return np.array(feats, dtype=np.float32)


def smiles_to_graph(
    smiles: str,
    max_atoms: int = MAX_ATOMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Convert SMILES string to padded (atom_feats, adj_norm, mask).

    The adjacency matrix is symmetrically normalised with self-loops:
        A_hat = D^{-0.5} (A + I) D^{-0.5}

    Returns:
        atom_feats: (max_atoms, D_ATOM) float32 — zero-padded
        adj_norm:   (max_atoms, max_atoms) float32 — normalised, zero-padded
        mask:       (max_atoms,) bool — True for real atoms
    Returns None if SMILES cannot be parsed or molecule exceeds max_atoms.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("rdkit required: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    if n > max_atoms:
        log.debug("Molecule too large (%d atoms > %d), skipping: %s", n, max_atoms, smiles[:50])
        return None

    # Build atom feature matrix
    atom_feats = np.zeros((max_atoms, D_ATOM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        atom_feats[i] = _atom_features(atom)

    # Build adjacency matrix and normalise
    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # Add self-loops and compute D^{-0.5} A_tilde D^{-0.5}
    a_tilde = adj + np.eye(n, dtype=np.float32)
    d = a_tilde.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    a_hat = (d_inv_sqrt[:, None] * a_tilde) * d_inv_sqrt[None, :]

    # Pad adjacency to max_atoms × max_atoms
    adj_norm = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    adj_norm[:n, :n] = a_hat

    # Mask: True for real atoms
    mask = np.zeros(max_atoms, dtype=bool)
    mask[:n] = True

    return atom_feats, adj_norm, mask


def build_drug_graphs(
    smiles_dict: dict[str, str | None],
    drug_to_idx: dict[str, int],
    max_atoms: int = MAX_ATOMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded graph arrays for all drugs.

    Drugs without valid SMILES receive all-zero arrays (mask=False everywhere),
    producing zero GNN embeddings — consistent with Phase 2 zero fingerprints.

    Returns:
        atom_feats: (n_drugs, max_atoms, D_ATOM) float32
        adj_norm:   (n_drugs, max_atoms, max_atoms) float32
        mask:       (n_drugs, max_atoms) bool
    """
    n_drugs = len(drug_to_idx)
    atom_feats = np.zeros((n_drugs, max_atoms, D_ATOM), dtype=np.float32)
    adj_norm = np.zeros((n_drugs, max_atoms, max_atoms), dtype=np.float32)
    mask = np.zeros((n_drugs, max_atoms), dtype=bool)

    n_ok = 0
    n_missing = 0
    n_parse_fail = 0

    for drug_name, idx in drug_to_idx.items():
        smiles = smiles_dict.get(drug_name)
        if smiles is None:
            n_missing += 1
            continue

        result = smiles_to_graph(smiles, max_atoms)
        if result is None:
            n_parse_fail += 1
            log.warning("Failed to parse graph for %s", drug_name)
            continue

        atom_feats[idx], adj_norm[idx], mask[idx] = result
        n_ok += 1

    log.info(
        "Drug graphs: %d ok, %d missing SMILES, %d parse failures / %d total",
        n_ok,
        n_missing,
        n_parse_fail,
        n_drugs,
    )
    return atom_feats, adj_norm, mask


def get_drug_graphs(
    drug_to_idx: dict[str, int],
    processed_dir: Path = PROCESSED_DIR,
    max_atoms: int = MAX_ATOMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load drug graph arrays from cache (.npz) or compute and save.

    Returns:
        atom_feats: (n_drugs, max_atoms, D_ATOM) float32
        adj_norm:   (n_drugs, max_atoms, max_atoms) float32
        mask:       (n_drugs, max_atoms) bool
    """
    graphs_path = processed_dir / "drug_graphs.npz"
    if graphs_path.exists():
        data = np.load(graphs_path)
        return data["atom_feats"], data["adj_norm"], data["mask"]

    smiles_path = processed_dir / "drug_smiles.json"
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"drug_smiles.json not found at {smiles_path}. "
            "Run src/data/drug_features.py first to fetch SMILES."
        )

    with smiles_path.open() as f:
        smiles_dict: dict[str, str | None] = json.load(f)

    atom_feats, adj_norm, mask = build_drug_graphs(smiles_dict, drug_to_idx, max_atoms)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(graphs_path, atom_feats=atom_feats, adj_norm=adj_norm, mask=mask)
    log.info("Saved drug graphs to %s", graphs_path)

    return atom_feats, adj_norm, mask


def main() -> None:
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    drug_df = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    drugs = sorted(drug_df["drug_name"].unique())
    drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(drugs)}

    graphs_path = PROCESSED_DIR / "drug_graphs.npz"
    if graphs_path.exists():
        graphs_path.unlink()

    atom_feats, adj_norm, mask = get_drug_graphs(drug_to_idx)

    n_drugs = len(drug_to_idx)
    n_ok = int(mask.any(axis=1).sum())
    print(f"atom_feats: {atom_feats.shape}, dtype={atom_feats.dtype}")
    print(f"adj_norm:   {adj_norm.shape}, dtype={adj_norm.dtype}")
    print(f"mask:       {mask.shape}, dtype={mask.dtype}")
    print(f"Drugs with at least one atom: {n_ok}/{n_drugs}")


if __name__ == "__main__":
    main()
