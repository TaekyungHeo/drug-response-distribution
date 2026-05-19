#!/bin/bash
#SBATCH --job-name=04cell_06_multiomics
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/04_cell_representation/02_representation_alternatives/06_multi_omics/logs/sbatch_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/02_representation_alternatives/06_multi_omics/logs
~/.local/bin/uv run python3 \
    experiments/04_cell_representation/02_representation_alternatives/06_multi_omics/jobs/run.py
echo "Job finished: $(date)"
