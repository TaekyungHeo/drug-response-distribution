#!/bin/bash
#SBATCH --job-name=05sol_within_moa
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=experiments/05_solutions/02_training_distribution/01_within_moa/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/05_solutions/02_training_distribution/01_within_moa/logs

~/.local/bin/uv run python3 \
    experiments/05_solutions/02_training_distribution/01_within_moa/jobs/run.py

echo "Job finished: $(date)"
