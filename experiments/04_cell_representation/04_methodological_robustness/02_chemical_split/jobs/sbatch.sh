#!/bin/bash
#SBATCH --job-name=04cell_11_chem
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=experiments/04_cell_representation/04_methodological_robustness/02_chemical_split/logs/sbatch_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/04_methodological_robustness/02_chemical_split/logs
~/.local/bin/uv run python3 experiments/04_cell_representation/04_methodological_robustness/02_chemical_split/jobs/run.py
echo "Job finished: $(date)"
