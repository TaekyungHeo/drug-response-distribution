#!/bin/bash
# 06_objective_axis: SLURM array job — one GPU job per fold.
# Submit: sbatch sbatch_array.sh
# After all jobs finish: sbatch --dependency=afterok:$ARRAY_JOB_ID sbatch_aggregate.sh
#
# Runtime per fold: ≈ 3 h (3 conditions × 200 epochs per fold).
#SBATCH --job-name=03null_06_fold
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-9
#SBATCH --output=experiments/03_drug_feature_null/06_objective_axis/logs/sbatch_fold%a_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "Fold: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/06_objective_axis/logs
mkdir -p experiments/03_drug_feature_null/06_objective_axis/report/data

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/06_objective_axis/jobs/run.py \
    --fold "$SLURM_ARRAY_TASK_ID"

echo "=== Job finished: $(date) ==="
