#!/bin/bash
#SBATCH --job-name=01gvp_step1
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/01_metric_decomposition/01_global_vs_perdrug/logs/step1_%j.log

set -euo pipefail
cd /home/spark/multi-onco

~/.local/bin/uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/src/step1_variance_decomposition.py
