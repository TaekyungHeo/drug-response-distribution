#!/usr/bin/env bash
#SBATCH --job-name=baselines_oracles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --nodelist=spark1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.out
#SBATCH --error=experiments/01_metric_decomposition/03_baselines/logs/%j_%x.err

set -euo pipefail

cd /home/spark/multi-onco
git fetch origin
git reset --hard origin/main

SCRIPT="experiments/01_metric_decomposition/03_baselines/src/compute_oracles.py"

echo "[$(date)] Smoke test..."
~/.local/bin/uv run python3 "$SCRIPT" --smoke
echo "[$(date)] Smoke test passed."

echo "[$(date)] Running Stage 0: analytical oracles..."
~/.local/bin/uv run python3 "$SCRIPT"
echo "[$(date)] Done."
