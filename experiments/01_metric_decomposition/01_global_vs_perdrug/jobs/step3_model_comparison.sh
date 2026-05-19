#!/bin/bash
#SBATCH --job-name=01gvp_step3
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=experiments/01_metric_decomposition/01_global_vs_perdrug/logs/step3_%j.log

set -euo pipefail
cd /home/spark/multi-onco

~/.local/bin/uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/src/step3_model_comparison.py --device cuda
