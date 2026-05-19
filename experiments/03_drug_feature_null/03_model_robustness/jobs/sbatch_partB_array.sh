#!/bin/bash
# Part B: SLURM array — one GPU job per fold (GNN training + embedding extraction).
# Submit:
#   ARRAY_JOB=$(sbatch --parsable sbatch_partB_array.sh)
#   sbatch --dependency=afterok:$ARRAY_JOB sbatch_partB_aggregate.sh
#
# Runtime per fold: ≈ 3–4 h (200 epochs × GNN).
#SBATCH --job-name=03null_pB_fold
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --array=0-9
#SBATCH --output=experiments/03_drug_feature_null/03_model_robustness/logs/sbatch_partB_fold%a_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "Fold: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/03_model_robustness/logs
mkdir -p experiments/03_drug_feature_null/03_model_robustness/checkpoints
mkdir -p experiments/03_drug_feature_null/03_model_robustness/report/data

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/03_model_robustness/jobs/run_partB.py \
    --fold "$SLURM_ARRAY_TASK_ID"

echo "=== Job finished: $(date) ==="
