#!/bin/bash
#SBATCH --job-name=04cell_08_omics_all
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=experiments/04_cell_representation/02_representation_alternatives/06_multi_omics/logs/sbatch_omics_rna_mut_all_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/02_representation_alternatives/06_multi_omics/logs
~/.local/bin/uv run python3 experiments/04_cell_representation/02_representation_alternatives/06_multi_omics/jobs/run.py --condition rna_mut_all
echo "Job finished: $(date)"
