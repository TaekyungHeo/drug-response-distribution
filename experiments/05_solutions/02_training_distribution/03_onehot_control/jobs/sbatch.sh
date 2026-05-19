#!/bin/bash
#SBATCH --job-name=05sol_onehot
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=experiments/05_solutions/02_training_distribution/03_onehot_control/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/05_solutions/02_training_distribution/03_onehot_control/logs

~/.local/bin/uv run python3 \
    experiments/05_solutions/02_training_distribution/03_onehot_control/jobs/run.py

echo "Job finished: $(date)"
