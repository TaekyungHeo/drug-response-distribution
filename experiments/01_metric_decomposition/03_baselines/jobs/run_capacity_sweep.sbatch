#!/usr/bin/env bash
#SBATCH --job-name=baselines_capacity_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --nodelist=spark1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.out
#SBATCH --error=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.err
# Memory: RNA pair matrix float32 ≈ 12 GB; z-score copies ≈ 2×; GPU model + activations ≈ 4 GB.
# 3 capacities × 9 HP configs + 3 capacities × 3 splits ≈ 36 training runs (max_epochs=300).
# Estimated runtime on GB10: ~3 h.

set -euo pipefail

cd /home/spark/multi-onco
git fetch origin
git reset --hard origin/main

SCRIPT="experiments/01_metric_decomposition/03_baselines/src/run_capacity_sweep.py"

echo "[$(date)] Smoke test..."
~/.local/bin/uv run python3 "$SCRIPT" --smoke
echo "[$(date)] Smoke test passed."

echo "[$(date)] Running Stage 2A: capacity sweep..."
~/.local/bin/uv run python3 "$SCRIPT"
echo "[$(date)] Done."
