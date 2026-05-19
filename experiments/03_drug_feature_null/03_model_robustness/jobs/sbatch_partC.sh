#!/bin/bash
# Part C: Submit fold-sharded jobs for extended drug representation ablation.
# Splits 10 folds across available nodes for parallel execution.
#
# Usage:
#   # Submit all 10 folds as separate jobs (scheduler assigns nodes):
#   bash experiments/03_drug_feature_null/03_model_robustness/jobs/sbatch_partC.sh
#
#   # Or submit specific folds manually:
#   sbatch --export=FOLD=0 experiments/03_drug_feature_null/03_model_robustness/jobs/sbatch_partC_fold.sh
#   sbatch --export=FOLD=1 experiments/03_drug_feature_null/03_model_robustness/jobs/sbatch_partC_fold.sh

set -euo pipefail
cd /home/spark/multi-onco

echo "Submitting Part C: 10 fold-sharded jobs"
echo "Each job: 3 conditions × 200 epochs ≈ 3h on DGX Spark"

for FOLD in $(seq 0 9); do
    sbatch --export=FOLD=${FOLD} \
        experiments/03_drug_feature_null/03_model_robustness/jobs/sbatch_partC_fold.sh
    echo "  Submitted fold ${FOLD}"
done

echo ""
echo "After all jobs complete, run aggregation:"
echo "  uv run python3 experiments/03_drug_feature_null/03_model_robustness/jobs/aggregate_partC.py"
