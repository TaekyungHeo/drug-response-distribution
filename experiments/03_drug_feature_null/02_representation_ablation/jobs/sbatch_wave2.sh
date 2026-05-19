#!/bin/bash
# Wave 2: gnn condition only. Requires 03 Part B to have produced
# data/processed/gnn_embeddings_256.npy before submitting this job.
#SBATCH --job-name=03null_02_ablation_gnn
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=experiments/03_drug_feature_null/02_representation_ablation/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

if [ ! -f data/processed/gnn_embeddings_256.npy ]; then
    echo "ERROR: data/processed/gnn_embeddings_256.npy not found." >&2
    echo "Run 03_model_robustness Part B first." >&2
    exit 1
fi

mkdir -p experiments/03_drug_feature_null/02_representation_ablation/logs

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/02_representation_ablation/jobs/run.py \
    --only gnn

echo "Job finished: $(date)"
