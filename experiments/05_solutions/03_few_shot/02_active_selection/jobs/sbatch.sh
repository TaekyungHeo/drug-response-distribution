#!/bin/bash
# Active cell selection for K-shot response matching.
# CPU only, <1h.
#SBATCH --job-name=05sol_active
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/05_solutions/03_few_shot/02_active_selection/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/05_solutions/03_few_shot/02_active_selection/logs

~/.local/bin/uv run python3 \
    experiments/05_solutions/03_few_shot/02_active_selection/jobs/run.py

echo "Job finished: $(date)"
