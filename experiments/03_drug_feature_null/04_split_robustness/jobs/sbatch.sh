#!/bin/bash
# Scaffold-blind 5-fold Ridge ablation (morgan_fp vs no_drug).
# Runtime < 15 min (CPU, Ridge only, 2 conditions × 5 folds).
# Memory: 128K × 2798 features × float64 ≈ 3 GB peak. --mem=16G has 5× headroom.
#SBATCH --job-name=03null_04_scaffold
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=experiments/03_drug_feature_null/04_split_robustness/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/04_split_robustness/logs

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/04_split_robustness/jobs/run.py

echo "Job finished: $(date)"
