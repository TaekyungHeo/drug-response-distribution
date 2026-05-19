#!/bin/bash
#SBATCH --job-name=05sol_lincs_moa
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=experiments/05_solutions/05_combinations/02_lincs_x_moa/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/05_solutions/05_combinations/02_lincs_x_moa/logs

~/.local/bin/uv run python3 \
    experiments/05_solutions/05_combinations/02_lincs_x_moa/jobs/run.py

echo "Job finished: $(date)"
