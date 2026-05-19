# 02 — Decompose PASO's r = 0.745

Quantifies how much of PASO's headline figure is explained by two protocol artifacts.

| Artifact | Contribution |
|----------|-------------|
| Test-set snooping (fair mean → PASO-style mean) | +0.042 |
| Best-fold cherry-picking (PASO-style mean → best fold) | +0.200 |
| Total inflation (fair mean → best reproduced fold, r=0.751 ≈ PASO's 0.745) | +0.242 |

Primary metric: **global Pearson r** for apples-to-apples comparison with PASO.
Per-drug r is also computed at each checkpoint to show metric inflation separately.

## Quick start

```bash
# 1. Prerequisites (PASO splits + processed omics)
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init external/PASO
# run experiments/00_data_preparation/jobs/sbatch.sh if data/processed/ is empty

# 2. Submit
cd ~/multi-onco
sbatch experiments/02_reproductions/01_paso/02_decomposition/jobs/sbatch.sh

# 3. After run completes, generate metrics.json
uv run python experiments/02_reproductions/01_paso/02_decomposition/metrics.py
```

Runtime: ~6–10 h on one GPU.

## How it works

Two conditions run side-by-side on PASO's fixed 10-fold drug-blind splits using
TransformerEncoder (Morgan FP + RNA + mutations; referred to as OmniCancerV1 internally):

**PASO-style**: test-set Pearson is monitored after every epoch; the checkpoint with
the highest test r across all 200 epochs is the final model. This is PASO's actual
protocol — test data directly drives model selection.

**Fair**: 10% of training drugs are held out entirely as validation (drug-blind, so
val and test are on the same distribution). Checkpoint selected by best val r.
Test set is never seen during training or selection.

See [PLAN.md](PLAN.md) for design rationale, prerequisites, and expected values.
