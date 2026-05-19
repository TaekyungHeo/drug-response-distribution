#!/bin/bash
# 02_moa_ceiling: Within-MoA pairwise profile concordance.
# CPU only, pure data analysis (no model training).
#SBATCH --job-name=05sol_diag_ceil
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=experiments/05_solutions/01_diagnosis/02_moa_ceiling/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/05_solutions/01_diagnosis/02_moa_ceiling/logs

~/.local/bin/uv run python3 \
    experiments/05_solutions/01_diagnosis/02_moa_ceiling/jobs/run.py

echo "Job finished: $(date)"
