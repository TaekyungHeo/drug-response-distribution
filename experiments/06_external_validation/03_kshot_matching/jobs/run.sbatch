#!/usr/bin/env bash
#SBATCH --job-name=06_kshot_matching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --nodelist=spark2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=experiments/06_external_validation/03_kshot_matching/logs/%j_%x.out
#SBATCH --error=experiments/06_external_validation/03_kshot_matching/logs/%j_%x.err

set -euo pipefail

cd /home/spark/multi-onco
git fetch origin
git reset --hard origin/main || git reset --hard --no-recurse-submodules origin/main
~/.local/bin/uv sync --index-strategy unsafe-best-match

SCRIPT="experiments/06_external_validation/03_kshot_matching/jobs/run.py"
PYTHON=".venv/bin/python3"

echo "[$(date)] Smoke test..."
$PYTHON -u "$SCRIPT" --smoke
echo "[$(date)] Smoke test passed."

echo "[$(date)] Running 03_kshot_matching (CTRPv2 + BeatAML + PRISM)..."
$PYTHON -u "$SCRIPT"
echo "[$(date)] Done."
