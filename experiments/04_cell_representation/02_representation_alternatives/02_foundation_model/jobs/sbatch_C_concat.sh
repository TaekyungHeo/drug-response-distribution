#!/bin/bash
#SBATCH --job-name=04cell_06_scf_C
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/04_cell_representation/02_representation_alternatives/04_foundation_model/logs/sbatch_scf_C_concat_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/02_representation_alternatives/04_foundation_model/logs
~/.local/bin/uv run python3 experiments/04_cell_representation/02_representation_alternatives/04_foundation_model/jobs/run.py --condition C_concat
echo "Job finished: $(date)"
