#!/bin/bash
# PRISM Ridge ablation (morgan_fp vs no_drug, 657 drugs × 477 cells).
# Runtime < 30 min (CPU, Ridge only, 2 conditions × 5 folds).
# Memory: 250K train pairs × 2798 features × float64 ≈ 5.6 GB peak. --mem=32G safe.
#SBATCH --job-name=03null_05_prism_ridge
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=experiments/03_drug_feature_null/05_dataset_robustness/logs/sbatch_ridge_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/05_dataset_robustness/logs

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/05_dataset_robustness/jobs/run_ridge.py

echo "Job finished: $(date)"
