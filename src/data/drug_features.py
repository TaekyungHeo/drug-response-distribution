"""Fetch drug SMILES from PubChem (+ ChEMBL fallback) and compute Morgan fingerprints."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

log = logging.getLogger(__name__)


def _strip_annotations(name: str) -> str:
    """Remove parenthetical concentration/stereochemistry annotations from drug names.

    Examples:
        "Bleomycin (50 uM)"  -> "Bleomycin"
        "Nutlin-3a (-)"      -> "Nutlin-3a"
        "GSK-LSD1-2HCl "     -> "GSK-LSD1-2HCl"
    """
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return cleaned or name.strip()


def _pubchem_lookup(name: str) -> str | None:
    """Single PubChem name lookup. Returns SMILES or None."""
    encoded = urllib.parse.quote(name)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/IsomericSMILES/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        return props.get("IsomericSMILES") or props.get("SMILES")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            log.debug("PubChem HTTP %d for: %s", exc.code, name)
    except Exception as exc:
        log.debug("PubChem error for %s: %s", name, exc)
    return None


def _chembl_lookup(name: str) -> str | None:
    """ChEMBL molecule lookup by preferred name. Returns SMILES or None."""
    encoded = urllib.parse.quote(name)
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule?pref_name__iexact={encoded}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        mols = data.get("molecules", [])
        if mols:
            smiles = mols[0].get("molecule_structures", {})
            return smiles.get("canonical_smiles") or smiles.get("standard_inchi")
    except Exception as exc:
        log.debug("ChEMBL error for %s: %s", name, exc)
    return None


def fetch_smiles(drug_names: list[str], cache_path: Path) -> dict[str, str | None]:
    """Fetch SMILES for drug names with PubChem → ChEMBL fallback, local JSON cache.

    Lookup chain for each drug:
    1. PubChem with original name
    2. PubChem with annotation-stripped name (removes "(50 uM)" etc.)
    3. ChEMBL with annotation-stripped name

    Args:
        drug_names: List of drug names to look up.
        cache_path: Path to JSON cache file (created/updated in place).

    Returns:
        Dict mapping drug_name -> SMILES string or None if not found.
    """
    cache: dict[str, str | None] = {}
    if cache_path.exists():
        with cache_path.open() as f:
            cache = json.load(f)

    to_fetch = [n for n in drug_names if n not in cache]
    log.info("Fetching SMILES for %d drugs (%d cached)", len(to_fetch), len(cache))

    for i, name in enumerate(to_fetch):
        smiles: str | None = None

        # 1. PubChem — original name
        smiles = _pubchem_lookup(name)
        time.sleep(0.25)

        # 2. PubChem — stripped name (if different)
        if smiles is None:
            stripped = _strip_annotations(name)
            if stripped != name:
                smiles = _pubchem_lookup(stripped)
                time.sleep(0.25)

        # 3. ChEMBL fallback
        if smiles is None:
            stripped = _strip_annotations(name)
            smiles = _chembl_lookup(stripped)
            time.sleep(0.25)

        if smiles is None:
            log.warning("No SMILES found for drug (using zeros): %s", name)
        cache[name] = smiles

        if (i + 1) % 10 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w") as f:
                json.dump(cache, f, indent=2)
            log.info("Saved cache after %d new fetches", i + 1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(cache, f, indent=2)

    return cache


def compute_fingerprints(
    smiles_dict: dict[str, str | None],
    drug_to_idx: dict[str, int],
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints (radius=2) for each drug.

    Args:
        smiles_dict: Mapping drug_name -> SMILES or None.
        drug_to_idx: Mapping drug_name -> integer index.
        n_bits: Fingerprint bit width.

    Returns:
        float32 array of shape (len(drug_to_idx), n_bits).
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("rdkit required: pip install rdkit")

    n_drugs = len(drug_to_idx)
    fps = np.zeros((n_drugs, n_bits), dtype=np.float32)

    for drug_name, idx in drug_to_idx.items():
        smiles = smiles_dict.get(drug_name)
        if smiles is None:
            log.warning("No SMILES for drug (using zeros): %s", drug_name)
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning("RDKit failed to parse SMILES for %s (using zeros)", drug_name)
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        fps[idx] = np.array(fp, dtype=np.float32)

    return fps


def get_drug_fingerprints(
    drug_to_idx: dict[str, int],
    processed_dir: Path = PROCESSED_DIR,
) -> np.ndarray:
    """Load drug fingerprints from cache or compute and save them.

    Args:
        drug_to_idx: Mapping drug_name -> integer index.
        processed_dir: Directory containing processed data files.

    Returns:
        float32 array of shape (len(drug_to_idx), 2048).
    """
    n_drugs = len(drug_to_idx)
    # Use a size-specific filename so different drug sets don't clobber each other.
    fp_path = processed_dir / f"drug_fingerprints_{n_drugs}.npy"
    legacy_path = processed_dir / "drug_fingerprints.npy"

    # Check size-specific cache first, then legacy cache (if row count matches).
    for candidate in (fp_path, legacy_path):
        if candidate.exists():
            arr = np.load(candidate)
            if arr.shape[0] == n_drugs:
                log.info("Loading cached fingerprints from %s", candidate.name)
                return arr

    drug_names = list(drug_to_idx.keys())
    cache_path = processed_dir / "drug_smiles.json"
    smiles_dict = fetch_smiles(drug_names, cache_path)

    fps = compute_fingerprints(smiles_dict, drug_to_idx)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(fp_path, fps)
    log.info("Saved fingerprints to %s", fp_path)

    return fps


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    drug_df = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    drugs = sorted(drug_df["drug_name"].unique())
    drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(drugs)}
    print(f"Drugs: {len(drugs)}")

    # Force recompute: remove existing file if present
    fp_path = PROCESSED_DIR / "drug_fingerprints.npy"
    if fp_path.exists():
        fp_path.unlink()

    fps = get_drug_fingerprints(drug_to_idx)

    smiles_path = PROCESSED_DIR / "drug_smiles.json"
    with smiles_path.open() as f:
        smiles_dict: dict[str, str | None] = json.load(f)

    n_valid = sum(1 for s in smiles_dict.values() if s is not None)
    n_total = len(drug_to_idx)
    n_nonzero_rows = int((fps.sum(axis=1) > 0).sum())

    print(f"Shape: {fps.shape}, dtype: {fps.dtype}")
    print(f"SMILES coverage: {n_valid}/{n_total} drugs ({100 * n_valid / n_total:.1f}%)")
    print(f"Non-zero fingerprint rows: {n_nonzero_rows}/{n_total}")

    missing = [d for d in drug_to_idx if smiles_dict.get(d) is None]
    if missing:
        print(
            f"Missing SMILES ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )


if __name__ == "__main__":
    main()
