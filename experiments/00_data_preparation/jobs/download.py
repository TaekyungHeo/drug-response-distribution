"""Download CCLE/DepMap and GDSC data files to data/raw/.

Sources:
  - DepMap 24Q4  (figshare, DOI 10.6084/m9.figshare.22765112)
  - CCLE legacy  (Broad Institute public FTP, accessed 2024)
  - GDSC2 release 8.4  (Sanger FTP, 24 Jul 2022 fit)

Usage:
    uv run python3 experiments/00_data_preparation/jobs/download.py [--force]
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=Warning, module="urllib3")

import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"

FILES: list[dict] = [
    # ── DepMap 24Q4 (figshare) ─────────────────────────────────────────────
    {
        "name": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "url": "https://ndownloader.figshare.com/files/51065489",
        "description": "RNA-seq gene expression (TPM log1p), DepMap 24Q4",
        "size_mb_approx": 250,
    },
    {
        "name": "OmicsSomaticMutations.csv",
        "url": "https://ndownloader.figshare.com/files/51065732",
        "description": "Somatic mutations, DepMap 24Q4",
        "size_mb_approx": 800,
    },
    {
        "name": "OmicsCNGene.csv",
        "url": "https://ndownloader.figshare.com/files/51065324",
        "description": "Copy number variation (log2 relative), DepMap 24Q4",
        "size_mb_approx": 400,
    },
    {
        "name": "Model.csv",
        "url": "https://ndownloader.figshare.com/files/51065297",
        "description": "Cell line metadata / sample info, DepMap 24Q4",
        "size_mb_approx": 5,
    },
    # ── CCLE legacy (Broad) ───────────────────────────────────────────────
    {
        "name": "CCLE_metabolomics_20190502.csv",
        "url": "https://data.broadinstitute.org/ccle/CCLE_metabolomics_20190502.csv",
        "description": "Metabolomics (225 metabolites, log10), CCLE 2019",
        "size_mb_approx": 2,
    },
    {
        "name": "CCLE_RPPA_20181003.csv",
        "url": "https://data.broadinstitute.org/ccle/CCLE_RPPA_20181003.csv",
        "description": "RPPA protein expression (214 antibodies), CCLE 2018",
        "size_mb_approx": 1,
    },
    # ── GDSC2 release 8.4 (Sanger) ───────────────────────────────────────
    {
        "name": "GDSC2_fitted_dose_response_24Jul22.csv",
        "url": "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.4/GDSC2_fitted_dose_response_24Jul22.csv",
        "description": "GDSC2 drug response IC50, release 8.4",
        "size_mb_approx": 30,
    },
]


def _download_file(url: str, dest: Path) -> None:
    response = requests.get(url, stream=True, allow_redirects=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar,
    ):
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))


def download_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_mb = sum(f["size_mb_approx"] for f in FILES)
    print(f"Destination: {RAW_DIR}")
    print(f"Total download ~{total_mb:,} MB\n")

    for file_info in FILES:
        dest = RAW_DIR / file_info["name"]
        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1e6
            print(f"  skip  {file_info['name']} ({size_mb:.1f} MB already present)")
            continue
        print(f"  fetch {file_info['name']}  —  {file_info['description']}")
        try:
            _download_file(file_info["url"], dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"  done  {file_info['name']} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  FAIL  {file_info['name']}: {e}", file=sys.stderr)
            if dest.exists():
                dest.unlink()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download raw CCLE/DepMap and GDSC data.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = parser.parse_args()
    download_all(force=args.force)
    print("\nAll files downloaded.")
