#!/bin/bash
#SBATCH --job-name=05sol_moa_kshot
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=experiments/05_solutions/05_combinations/01_moa_x_kshot/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/05_solutions/05_combinations/01_moa_x_kshot/logs

~/.local/bin/uv run python3 \
    experiments/05_solutions/05_combinations/01_moa_x_kshot/jobs/run.py

echo "Job finished: $(date)"
