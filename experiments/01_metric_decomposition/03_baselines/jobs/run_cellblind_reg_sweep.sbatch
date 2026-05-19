#!/usr/bin/env bash
#SBATCH --job-name=baselines_cb_reg_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --nodelist=spark1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.out
#SBATCH --error=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.err
# Memory: RNA pair matrix float32 ≈ 12 GB; z-score copies ≈ 2×; GPU model ≈ 1 GB.
# 9 configs × 1 split (cell-blind), max_epochs=300, early stopping patience=30.
# Estimated runtime on GB10: ~30 min.

set -euo pipefail

cd /home/spark/multi-onco
git fetch origin
git reset --hard origin/main

SCRIPT="experiments/01_metric_decomposition/03_baselines/src/run_cellblind_reg_sweep.py"

echo "[$(date)] Smoke test..."
~/.local/bin/uv run python3 "$SCRIPT" --smoke
echo "[$(date)] Smoke test passed."

echo "[$(date)] Running Stage 3: cell-blind regularization sweep..."
~/.local/bin/uv run python3 "$SCRIPT"
echo "[$(date)] Done."
