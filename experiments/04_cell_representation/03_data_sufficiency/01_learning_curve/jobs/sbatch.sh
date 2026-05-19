#!/bin/bash
#SBATCH --job-name=04cell_lc_datasuff
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/04_cell_representation/03_data_sufficiency/01_learning_curve/logs/sbatch_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/03_data_sufficiency/01_learning_curve/logs
~/.local/bin/uv run python3 \
    experiments/04_cell_representation/03_data_sufficiency/01_learning_curve/jobs/run.py
echo "Job finished: $(date)"
