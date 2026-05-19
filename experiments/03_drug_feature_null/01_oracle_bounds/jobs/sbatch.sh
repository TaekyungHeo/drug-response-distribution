#!/bin/bash
#SBATCH --job-name=03null_01_oracle
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=experiments/03_drug_feature_null/01_oracle_bounds/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/01_oracle_bounds/logs

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/01_oracle_bounds/jobs/run.py

echo "Job finished: $(date)"
