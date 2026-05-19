#!/usr/bin/env bash
#SBATCH --job-name=baselines_modality
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --nodelist=spark1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.out
#SBATCH --error=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.err
# One job per variant. Submit 5 times with different VARIANT values:
#   sbatch --export=VARIANT=rna_only        run_modality_ablation.sbatch
#   sbatch --export=VARIANT=rna_mut         run_modality_ablation.sbatch
#   sbatch --export=VARIANT=rna_mut_cnv     run_modality_ablation.sbatch
#   sbatch --export=VARIANT=rna_mut_cnv_met run_modality_ablation.sbatch
#   sbatch --export=VARIANT=all_5_omics     run_modality_ablation.sbatch
#
# Memory: cell_mat is ~280 MB; peak = Xtr+Xvl+Xte slices ≈ 57 GB for all_5_omics.
# No pair_X materialized — indexing directly from cell_mat per split.

set -euo pipefail

cd /home/spark/multi-onco
git fetch origin
git reset --hard origin/main

SCRIPT="experiments/01_metric_decomposition/03_baselines/src/run_modality_ablation.py"
if [[ -z "${VARIANT:-}" ]]; then
    echo "ERROR: VARIANT must be set. Use --export=VARIANT=<name> when submitting." >&2
    exit 1
fi

echo "[$(date)] Smoke test (${VARIANT})..."
~/.local/bin/uv run python3 "$SCRIPT" --variant "$VARIANT" --smoke
echo "[$(date)] Smoke test passed."

echo "[$(date)] Running Stage 2B: modality ablation (${VARIANT})..."
~/.local/bin/uv run python3 "$SCRIPT" --variant "$VARIANT"
echo "[$(date)] Done."
