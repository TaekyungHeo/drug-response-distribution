#!/bin/bash
#SBATCH --job-name=02ms_step2_bootstrap
#SBATCH --partition=all
#SBATCH --output=experiments/01_metric_decomposition/02_metric_selection/logs/step2_bootstrap_%A_%a.log
#SBATCH --array=0-4%1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
cd /home/spark/multi-onco

git fetch origin
git reset --hard origin/main

~/.local/bin/uv run python3 experiments/01_metric_decomposition/02_metric_selection/src/compute_fold_metrics.py --fold "$SLURM_ARRAY_TASK_ID"
