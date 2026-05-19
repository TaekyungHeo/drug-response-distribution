#!/bin/bash
#SBATCH --job-name=01gvp_step2
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/01_metric_decomposition/01_global_vs_perdrug/logs/step2_%j.log
# Step 2B uses OmniCancerV1 predictions from report/data/results.json (already run).
# If re-running step3 first: sbatch --dependency=afterok:<step3_jobid> step2_baseline_dissociation.sbatch

set -euo pipefail
cd /home/spark/multi-onco

git fetch origin
git reset --hard origin/main

~/.local/bin/uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/src/step2_baseline_dissociation.py
