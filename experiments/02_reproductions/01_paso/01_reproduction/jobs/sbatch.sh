#!/bin/bash
#SBATCH --job-name=paso_reproduction
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=experiments/02_reproductions/01_paso/01_reproduction/logs/sbatch_%j.log

set -euo pipefail
cd /home/spark/multi-onco

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

git fetch origin
git reset --hard origin/main

# pytoda==1.1.3 requires rdkit-pypi (numpy 1.x ABI) and ipython (fastprogress dep).
# --python 3.11: rdkit-pypi has no cp312+ wheels.
# --with "numpy<2": rdkit-pypi compiled against numpy 1.x C API.
# --with "ipython": required by fastprogress (SmilesPE transitive dep).
~/.local/bin/uv run --python 3.11 \
    --with "pytoda==1.1.3" \
    --with "numpy<2" \
    --with "ipython" \
    python3 experiments/02_reproductions/01_paso/01_reproduction/jobs/run.py

echo "Job finished: $(date)"
