#!/bin/bash
# PRISM OmniCancerV1 Transformer ablation (morgan_fp vs no_drug).
# Runtime ≈ 2 h (2 conditions × 5 folds × 200 epochs; PRISM has 313K pairs/fold vs 160K for GDSC2).
# Memory: model 42 MB bf16, batch activations < 100 MB (larger batches than GDSC2).
# Peak GPU < 6 GB; --mem=64G covers unified system allocation on DGX Spark.
#SBATCH --job-name=03null_05_prism_transformer
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=experiments/03_drug_feature_null/05_dataset_robustness/logs/sbatch_transformer_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/05_dataset_robustness/logs

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/05_dataset_robustness/jobs/run_transformer.py

echo "Job finished: $(date)"
