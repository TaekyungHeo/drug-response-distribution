#!/bin/bash
#SBATCH --job-name=rank_loss
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=experiments/04_cell_representation/04_methodological_robustness/03_ranking_loss/logs/sbatch_%j.log
set -euo pipefail
cd /home/spark/multi-onco
echo "Job started: $(date)"
echo "Node: $(hostname)"
git fetch origin
git reset --hard origin/main
mkdir -p experiments/04_cell_representation/04_methodological_robustness/03_ranking_loss/logs
~/.local/bin/uv run python3 experiments/04_cell_representation/04_methodological_robustness/03_ranking_loss/jobs/run.py
echo "Job finished: $(date)"
