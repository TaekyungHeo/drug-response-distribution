#!/usr/bin/env bash
#SBATCH --job-name=baselines_ridge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --nodelist=spark1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.out
#SBATCH --error=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.err

set -euo pipefail

cd /home/spark/multi-onco
git fetch origin
git reset --hard origin/main

SCRIPT="experiments/01_metric_decomposition/03_baselines/src/run_ridge.py"

echo "[$(date)] Smoke test..."
~/.local/bin/uv run python3 -u "$SCRIPT" --smoke
echo "[$(date)] Smoke test passed."

echo "[$(date)] Running Stage 1: Ridge regression (5 folds × 3 splits)..."
~/.local/bin/uv run python3 -u "$SCRIPT"
echo "[$(date)] Done."
