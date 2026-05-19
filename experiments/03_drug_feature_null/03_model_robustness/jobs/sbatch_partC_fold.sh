#!/bin/bash
# Part C: Single-fold job. Called via sbatch --export=FOLD=X.
# Runtime per fold: ~3h (3 conditions × 200 epochs).
#SBATCH --job-name=03null_partC
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=experiments/03_drug_feature_null/03_model_robustness/logs/partC_fold%02x_%j.log

set -euo pipefail
cd /home/spark/multi-onco

FOLD=${FOLD:-0}

echo "=== Part C fold ${FOLD} ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/03_drug_feature_null/03_model_robustness/{logs,checkpoints,report/data}

/home/spark/multi-onco/.venv/bin/python3 \
    experiments/03_drug_feature_null/03_model_robustness/jobs/run_partC.py \
    --fold ${FOLD}

echo "Fold ${FOLD} finished: $(date)"
