#!/bin/bash
# Gate experiment: can LINCS signatures be predicted from Morgan fingerprints?
# Ridge LOO on ~104 drugs, CPU only, <5 min.
#SBATCH --job-name=05sol_lincs_pred
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=experiments/05_solutions/04_external_signatures/02_lincs_prediction/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"

git fetch origin
git reset --hard origin/main

mkdir -p experiments/05_solutions/04_external_signatures/02_lincs_prediction/logs

~/.local/bin/uv run python3 \
    experiments/05_solutions/04_external_signatures/02_lincs_prediction/jobs/run.py

echo "Job finished: $(date)"
