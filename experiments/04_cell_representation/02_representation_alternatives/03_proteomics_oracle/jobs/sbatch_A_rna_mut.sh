#!/bin/bash
#SBATCH --job-name=04cell_07_rppa_A
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/04_cell_representation/02_representation_alternatives/05_proteomics_oracle/logs/sbatch_rppa_A_rna_mut_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/02_representation_alternatives/05_proteomics_oracle/logs
~/.local/bin/uv run python3 experiments/04_cell_representation/02_representation_alternatives/05_proteomics_oracle/jobs/run.py --condition A_rna_mut
echo "Job finished: $(date)"
