#!/bin/bash
#SBATCH --job-name=paso_decomp
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=experiments/02_reproductions/01_paso/02_decomposition/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

~/.local/bin/uv run python3 \
    experiments/02_reproductions/01_paso/02_decomposition/jobs/run.py

~/.local/bin/uv run python3 \
    experiments/02_reproductions/01_paso/02_decomposition/metrics.py

echo "Job finished: $(date)"
