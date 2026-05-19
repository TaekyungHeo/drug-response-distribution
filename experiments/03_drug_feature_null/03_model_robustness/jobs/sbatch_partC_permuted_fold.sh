#!/bin/bash
# Part C permuted MoA: Single-fold job. Called via sbatch --export=FOLD=X.
# Runtime per fold: ~1h (1 condition × 200 epochs).
#SBATCH --job-name=03null_perm
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=experiments/03_drug_feature_null/03_model_robustness/logs/partC_perm_%j.log

set -euo pipefail
cd /home/spark/multi-onco

FOLD=${FOLD:-0}

echo "=== Part C permuted MoA fold ${FOLD} ==="
echo "Started: $(date)"
echo "Node: $(hostname)"

mkdir -p experiments/03_drug_feature_null/03_model_robustness/{logs,checkpoints,report/data}

/home/spark/multi-onco/.venv/bin/python3 \
    experiments/03_drug_feature_null/03_model_robustness/jobs/run_partC.py \
    --fold ${FOLD} \
    --conditions moa_permuted

echo "Fold ${FOLD} finished: $(date)"
