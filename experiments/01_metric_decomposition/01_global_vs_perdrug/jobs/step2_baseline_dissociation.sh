#!/bin/bash
#SBATCH --job-name=01gvp_step2
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/01_metric_decomposition/01_global_vs_perdrug/logs/step2_%j.log

# Step 2B requires step3 predictions. Submit with --dependency=afterok:<step3_jobid>
set -euo pipefail
cd /home/spark/multi-onco

~/.local/bin/uv run python3 experiments/01_metric_decomposition/01_global_vs_perdrug/src/step2_baseline_dissociation.py
