#!/bin/bash
# Quick validation that all 11 experiments run without error on small data.
# Usage: bash experiments/05_solutions/smoke_test.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXPS=(
  01_diagnosis/01_moa_performance
  01_diagnosis/02_moa_ceiling
  02_training_distribution/01_within_moa
  02_training_distribution/02_moa_weighted
  02_training_distribution/03_onehot_control
  03_few_shot/01_response_matching
  03_few_shot/02_active_selection
  04_external_signatures/01_lincs
  04_external_signatures/02_lincs_prediction
  05_combinations/01_moa_x_kshot
  05_combinations/02_lincs_x_moa
)

PASS=0
FAIL=0

for exp in "${EXPS[@]}"; do
  echo "=== $exp ==="
  SCRIPT="experiments/05_solutions/$exp/jobs/run.py"
  if [[ ! -f "$SCRIPT" ]]; then
    echo "  SKIP: $SCRIPT not found"
    continue
  fi
  if uv run python3 -u "$SCRIPT" --smoke 2>&1; then
    echo "  PASS"
    ((PASS++))
  else
    echo "  FAIL"
    ((FAIL++))
  fi
  echo ""
done

echo "=== Summary: $PASS passed, $FAIL failed out of ${#EXPS[@]} ==="
[[ $FAIL -eq 0 ]] && echo "All smoke tests passed." || exit 1
