#!/bin/bash
# Part B: 10-fold OmniCancerV2 GCN training + embedding extraction.
# Each fold trains on fold-k train drugs and extracts embeddings for fold-k test drugs,
# so every drug is embedded by a checkpoint that never saw it in training.
# Runtime ≈ 20 h (10 folds × ≈2 h/fold).
# Output: data/processed/gnn_embeddings_256.npy (233 drugs × 256 dims).
# Memory: GCN params < 5 MB bf16, drug graph batch < 50 MB. Peak GPU < 1 GB.
#SBATCH --job-name=03null_03_partB
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=experiments/03_drug_feature_null/03_model_robustness/logs/sbatch_partB_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

if [ ! -f data/processed/drug_graphs.npz ]; then
    echo "ERROR: data/processed/drug_graphs.npz not found." >&2
    exit 1
fi

mkdir -p experiments/03_drug_feature_null/03_model_robustness/logs
mkdir -p experiments/03_drug_feature_null/03_model_robustness/checkpoints
mkdir -p data/processed

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/03_model_robustness/jobs/run_partB.py

echo "Output: $(python3 -c "import numpy as np; x=np.load('data/processed/gnn_embeddings_256.npy'); print(x.shape)")"
echo "Job finished: $(date)"
