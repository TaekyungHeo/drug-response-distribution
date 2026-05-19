#!/bin/bash
# Part A: 10-fold OmniCancerV1 Transformer ablation (morgan_fp vs no_drug).
# Runtime ≈ 20 h (2 conditions × 10 folds × 200 epochs).
# Memory: model 42 MB bf16, batch activations < 50 MB, RNA data 52 MB (CPU).
# Peak GPU < 1 GB; --mem=64G covers unified system allocation on DGX Spark.
#SBATCH --job-name=03null_03_partA
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=experiments/03_drug_feature_null/03_model_robustness/logs/sbatch_partA_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/03_model_robustness/logs
mkdir -p experiments/03_drug_feature_null/03_model_robustness/checkpoints

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/03_model_robustness/jobs/run_partA.py

echo "Job finished: $(date)"
