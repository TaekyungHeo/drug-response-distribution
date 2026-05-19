#!/usr/bin/env bash
#SBATCH --job-name=06_drug_feature_null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --nodelist=spark1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=experiments/06_external_validation/01_drug_feature_null/logs/%j_%x.out
#SBATCH --error=experiments/06_external_validation/01_drug_feature_null/logs/%j_%x.err

set -euo pipefail

cd /home/spark/multi-onco
git fetch origin
git reset --hard origin/main

SCRIPT="experiments/06_external_validation/01_drug_feature_null/jobs/run.py"

echo "[$(date)] Smoke test..."
~/.local/bin/uv run python3 -u "$SCRIPT" --smoke
echo "[$(date)] Smoke test passed."

echo "[$(date)] Running 01_drug_feature_null (CTRPv2 + BeatAML + PRISM)..."
~/.local/bin/uv run python3 -u "$SCRIPT"
echo "[$(date)] Done."
