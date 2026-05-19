#!/bin/bash
#SBATCH --job-name=04cell_05_repr_pathway
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/04_cell_representation/02_representation_alternatives/03_representation_sweep/logs/sbatch_repr_pathway_kegg_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/02_representation_alternatives/03_representation_sweep/logs
~/.local/bin/uv run python3 experiments/04_cell_representation/02_representation_alternatives/03_representation_sweep/jobs/run.py --condition pathway_kegg
echo "Job finished: $(date)"
