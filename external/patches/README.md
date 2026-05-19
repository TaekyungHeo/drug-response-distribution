# Patches for External Code Repositories

This directory holds `.patch` files that document modifications to upstream code
vendored in `external/`.

## Current status

All three vendored directories are **clean copies** — no upstream code has been modified.
Patches will be added here if and when modifications are needed.

| Directory | Upstream URL | Pinned commit | Patch file |
|-----------|-------------|---------------|------------|
| `external/DeepCDR` | https://github.com/kimmo1019/DeepCDR.git | `4dc5a90` | — |
| `external/DrugCell` | https://github.com/idekerlab/DrugCell.git | `c507e1d` | — |
| `external/PASO` | https://github.com/queryang/PASO.git | `8a7a4ce` | — |

## How to create a patch

```bash
# From the repo root, diff the vendored directory against the pinned upstream commit
git diff <pinned-commit> -- external/PASO/ > external/patches/PASO.patch

# Document what was changed and why at the top of the patch file
# (add a comment block before the diff output)
```

## How to apply a patch

```bash
cd external/PASO
git apply ../patches/PASO.patch
```

## Patch file format

Each patch file should start with a comment block explaining the motivation:

```
# PATCH: PASO — fix data loading path for cross-platform compatibility
# Upstream issue: <URL if applicable>
# Applied to commit: 8a7a4ce
# ---
<git diff output>
```
