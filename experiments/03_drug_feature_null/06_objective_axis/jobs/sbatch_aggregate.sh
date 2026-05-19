#!/bin/bash
# Aggregate 06_objective_axis fold shards into metrics.json.
# Run after all array jobs complete:
#   ARRAY_JOB=$(sbatch --parsable sbatch_array.sh)
#   sbatch --dependency=afterok:$ARRAY_JOB sbatch_aggregate.sh
#
#SBATCH --job-name=03null_06_agg
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=experiments/03_drug_feature_null/06_objective_axis/logs/sbatch_aggregate_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "=== Aggregation started: $(date) ==="
git fetch origin
git reset --hard origin/main

~/.local/bin/uv run python3 \
    experiments/03_drug_feature_null/06_objective_axis/jobs/aggregate.py

echo "=== Aggregation finished: $(date) ==="
