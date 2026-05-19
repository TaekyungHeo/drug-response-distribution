"""Fetch drug-protein target features from ChEMBL REST API.

from __future__ import annotations

Each drug is mapped to a binary vector over known protein targets.
Drugs without ChEMBL entries receive zero vectors (same fallback as fingerprints).

Output: data/processed/drug_target_features.npy  shape=(n_drugs, n_targets)
        data/processed/drug_target_index.json     target UniProt ID → column index
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

log = logging.getLogger(__name__)


def _get(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    """HTTP GET with retries, returns parsed JSON."""
    import urllib.error
    import urllib.request

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return {}
            if attempt == retries - 1:
                return {}
            time.sleep(delay * (attempt + 1))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
    return {}


def search_chembl_id(drug_name: str) -> str | None:
    """Search ChEMBL for a drug name, return best chembl_id or None."""
    import urllib.parse

    encoded = urllib.parse.quote(drug_name, safe="")
    url = f"{CHEMBL_API}/molecule.json?pref_name__iexact={encoded}&limit=1"
    data = _get(url)
    molecules = data.get("molecules", [])
    if molecules:
        return molecules[0].get("molecule_chembl_id")

    # Fallback: synonym search
    url2 = f"{CHEMBL_API}/molecule.json?molecule_synonyms__synonym__iexact={encoded}&limit=1"
    data2 = _get(url2)
    molecules2 = data2.get("molecules", [])
    if molecules2:
        return molecules2[0].get("molecule_chembl_id")
    return None


def get_targets_for_chembl_id(chembl_id: str, activity_type: str = "IC50") -> set[str]:
    """Return set of UniProt accession IDs targeted by a ChEMBL compound."""
    targets: set[str] = set()
    offset = 0
    limit = 100
    while True:
        url = (
            f"{CHEMBL_API}/activity.json?molecule_chembl_id={chembl_id}"
            f"&standard_type={activity_type}&limit={limit}&offset={offset}"
        )
        data = _get(url)
        activities = data.get("activities", [])
        for act in activities:
            tid = act.get("target_chembl_id")
            if not tid:
                continue
            # Fetch UniProt for this target
            t_url = f"{CHEMBL_API}/target/{tid}.json"
            t_data = _get(t_url)
            for comp in t_data.get("target_components", []):
                for xref in comp.get("target_component_xrefs", []):
                    if xref.get("xref_src_db") == "UniProt":
                        targets.add(xref["xref_id"])
        if len(activities) < limit:
            break
        offset += limit
    return targets


def compute_drug_target_features(
    drug_to_idx: dict[str, int],
    cache_path: Path | None = None,
) -> np.ndarray:
    """Build binary drug × target feature matrix using ChEMBL API.

    Args:
        drug_to_idx: Mapping drug_name → integer index.
        cache_path: If set, save/load per-drug target sets as JSON.

    Returns:
        float32 array of shape (n_drugs, n_unique_targets).
    """
    n_drugs = len(drug_to_idx)

    # Load or build per-drug target sets
    target_sets: dict[str, set[str]] = {}
    drug_chembl_ids: dict[str, str | None] = {}

    cache_data: dict = {}
    if cache_path and cache_path.exists():
        with cache_path.open() as f:
            cache_data = json.load(f)
        log.info("Loaded target cache for %d drugs", len(cache_data))

    for drug_name in drug_to_idx:
        if drug_name in cache_data:
            chembl_id = cache_data[drug_name].get("chembl_id")
            targets = set(cache_data[drug_name].get("targets", []))
        else:
            log.info("  Fetching ChEMBL targets for: %s", drug_name)
            chembl_id = search_chembl_id(drug_name)
            targets: set[str] = set()
            if chembl_id:
                targets = get_targets_for_chembl_id(chembl_id)
            cache_data[drug_name] = {"chembl_id": chembl_id, "targets": list(targets)}
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("w") as f:
                    json.dump(cache_data, f, indent=2)
            time.sleep(0.3)  # rate-limit

        drug_chembl_ids[drug_name] = chembl_id
        target_sets[drug_name] = targets

    # Build target vocabulary
    all_targets: list[str] = sorted(set().union(*[s for s in target_sets.values() if s]))
    target_to_idx = {t: i for i, t in enumerate(all_targets)}
    n_targets = len(all_targets)

    n_with_targets = sum(1 for s in target_sets.values() if s)
    log.info(
        "Drug target vocabulary: %d unique targets, %d/%d drugs have targets",
        n_targets,
        n_with_targets,
        n_drugs,
    )

    features = np.zeros((n_drugs, max(n_targets, 1)), dtype=np.float32)
    for drug_name, drug_idx in drug_to_idx.items():
        for target in target_sets.get(drug_name, set()):
            if target in target_to_idx:
                features[drug_idx, target_to_idx[target]] = 1.0

    return features, target_to_idx


def get_drug_target_features(
    drug_to_idx: dict[str, int],
    processed_dir: Path = PROCESSED_DIR,
    force_recompute: bool = False,
) -> np.ndarray:
    """Load drug target features from cache or compute via ChEMBL API."""
    feat_path = processed_dir / "drug_target_features.npy"
    idx_path = processed_dir / "drug_target_index.json"
    cache_path = processed_dir / "drug_target_cache.json"

    if feat_path.exists() and idx_path.exists() and not force_recompute:
        log.info("Loading cached drug target features from %s", feat_path)
        return np.load(feat_path)

    features, target_to_idx = compute_drug_target_features(drug_to_idx, cache_path=cache_path)

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, features)
    with idx_path.open("w") as f:
        json.dump(target_to_idx, f)
    log.info("Saved drug target features %s to %s", features.shape, feat_path)
    return features


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    drug_df = pd.read_parquet(PROCESSED_DIR / "drug_response.parquet")
    drugs = sorted(drug_df["drug_name"].unique())
    drug_to_idx = {d: i for i, d in enumerate(drugs)}
    features = get_drug_target_features(drug_to_idx, force_recompute=True)
    n_with = int((features.sum(axis=1) > 0).sum())
    print(f"Shape: {features.shape}, dtype: {features.dtype}")
    print(f"Drugs with targets: {n_with}/{len(drug_to_idx)}")
    print(f"Mean targets per drug: {features.sum(axis=1).mean():.1f}")
