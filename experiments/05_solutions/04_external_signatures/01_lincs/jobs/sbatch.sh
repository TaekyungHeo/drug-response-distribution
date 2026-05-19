#!/bin/bash
#SBATCH --job-name=05sol_lincs
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=experiments/05_solutions/04_external_signatures/01_lincs/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/05_solutions/04_external_signatures/01_lincs/logs

~/.local/bin/uv run python3 \
    experiments/05_solutions/04_external_signatures/01_lincs/jobs/run.py

echo "Job finished: $(date)"
