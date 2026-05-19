#!/bin/bash
#SBATCH --job-name=01gvp_step4
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=experiments/01_metric_decomposition/01_global_vs_perdrug/logs/step4_%j.log

set -euo pipefail
cd /home/spark/multi-onco

~/.local/bin/uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/src/step4_cross_split.py --device cuda
