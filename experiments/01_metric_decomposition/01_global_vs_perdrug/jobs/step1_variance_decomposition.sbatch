#!/bin/bash
#SBATCH --job-name=01gvp_step1
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/01_metric_decomposition/01_global_vs_perdrug/logs/step1_%j.log

set -euo pipefail
cd /home/spark/multi-onco

git fetch origin
git reset --hard origin/main

~/.local/bin/uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/src/step1_variance_decomposition.py
