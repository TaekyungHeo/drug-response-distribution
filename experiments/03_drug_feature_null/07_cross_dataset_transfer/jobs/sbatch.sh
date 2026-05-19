#!/bin/bash
# 07_cross_dataset_transfer: GDSC2 → PRISM Ridge ablation (no GPU needed).
# Submit: sbatch experiments/03_drug_feature_null/07_cross_dataset_transfer/jobs/sbatch.sh
#
#SBATCH --job-name=03null_07_xfer
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=experiments/03_drug_feature_null/07_cross_dataset_transfer/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/07_cross_dataset_transfer/logs
mkdir -p experiments/03_drug_feature_null/07_cross_dataset_transfer/report/data

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/07_cross_dataset_transfer/jobs/run.py

echo "=== Job finished: $(date) ==="
