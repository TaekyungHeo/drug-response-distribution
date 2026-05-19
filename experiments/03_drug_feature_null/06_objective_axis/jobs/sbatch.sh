#!/bin/bash
# MLP MSE vs RankNet ablation (3 conditions × 5 folds × 200 epochs).
# Runtime ≈ 5 h (GPU, 10 folds). RankNet uses streaming pair sampler — full pair matrix is
# never precomputed (55M pairs would be ~880 MB; streaming uses N_pairs=100/drug/step).
# Memory: MLP params < 10 MB, MSE batch 256 pairs × 2798 features < 10 MB,
# RankNet streaming buffer < 50 MB. Peak GPU << 2 GB; --mem=64G for system.
#SBATCH --job-name=03null_06_objective
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=experiments/03_drug_feature_null/06_objective_axis/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/06_objective_axis/logs

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/06_objective_axis/jobs/run.py

echo "Job finished: $(date)"
