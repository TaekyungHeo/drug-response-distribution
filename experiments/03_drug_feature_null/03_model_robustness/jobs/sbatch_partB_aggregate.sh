#!/bin/bash
# Aggregate partB fold shards. Run after array job:
#   sbatch --dependency=afterok:$ARRAY_JOB_ID sbatch_partB_aggregate.sh
#
#SBATCH --job-name=03null_pB_agg
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=experiments/03_drug_feature_null/03_model_robustness/logs/sbatch_partB_agg_%j.log

set -euo pipefail
cd /home/spark/multi-onco
git fetch origin && git reset --hard origin/main

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/03_model_robustness/jobs/aggregate_partB.py

echo "Done: $(date)"
