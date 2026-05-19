#!/bin/bash
#SBATCH --job-name=04cell_06_scf_B
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/04_cell_representation/02_representation_alternatives/04_foundation_model/logs/sbatch_scf_B_scfoundation_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/02_representation_alternatives/04_foundation_model/logs
~/.local/bin/uv run python3 experiments/04_cell_representation/02_representation_alternatives/04_foundation_model/jobs/run.py --condition B_scfoundation
echo "Job finished: $(date)"
