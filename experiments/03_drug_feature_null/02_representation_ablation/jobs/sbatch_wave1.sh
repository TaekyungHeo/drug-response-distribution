#!/bin/bash
# Wave 1: all conditions except gnn (no prerequisites).
# Submit immediately. Wave 2 (gnn) runs after 03 Part B finishes.
#SBATCH --job-name=03null_02_ablation_w1
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/03_drug_feature_null/02_representation_ablation/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/02_representation_ablation/logs

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/02_representation_ablation/jobs/run.py \
    --skip gnn

echo "Job finished: $(date)"
