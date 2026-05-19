#!/bin/bash
#SBATCH --job-name=02ms_step3_aggregate
#SBATCH --partition=all
#SBATCH --output=experiments/01_metric_decomposition/02_metric_selection/logs/step3_aggregate_%j.log
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd /home/spark/multi-onco

git fetch origin
git reset --hard origin/main

~/.local/bin/uv run python3 experiments/01_metric_decomposition/02_metric_selection/src/aggregate_results.py
