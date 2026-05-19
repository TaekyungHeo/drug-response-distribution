# External Dependencies

This directory contains prior-work baselines used for comparison. It is split into two
categories: **code repositories** (vendored directly into the repo) and **data files**
(stored under `data/external/`, not tracked in git).

---

## Code repositories

These directories are vendored copies of upstream repositories, not git submodules.
The pinned commit records the upstream revision used.

| Directory | Paper | Upstream | Pinned commit |
|-----------|-------|----------|---------------|
| `DeepCDR/` | Huang et al. 2020, *Bioinformatics* | https://github.com/kimmo1019/DeepCDR.git | `4dc5a90` |
| `DrugCell/` | Kuenzi et al. 2020, *Cancer Cell* | https://github.com/idekerlab/DrugCell.git | `c507e1d` |
| `PASO/` | Yang et al. 2023 | https://github.com/queryang/PASO.git | `8a7a4ce` |

### Patches

No upstream code has been modified. If modifications are needed in future, patch files
go in `external/patches/`. See [`patches/README.md`](patches/README.md).

---

## Data directories (not tracked in git)

Stored under `data/external/` which is covered by the root `.gitignore`.

| Directory | Source | Access |
|-----------|--------|--------|
| `data/external/beataml2/` | BeatAML (Bottomly et al. 2022) | NIH dbGaP phs001657 (controlled access) |
| `data/external/scFoundation/` | scFoundation embeddings (biomap-research) | Public, requires download |

For data acquisition instructions see:
- BeatAML: [`experiments/08_external_validation/03_beataml_validation/report/README.md`](../experiments/08_external_validation/03_beataml_validation/report/README.md)
- scFoundation: [`experiments/04_cell_representation/06_scfoundation/README.md`](../experiments/04_cell_representation/06_scfoundation/README.md)
