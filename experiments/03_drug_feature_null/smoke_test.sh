#!/bin/bash
# Smoke test: verify all 03_drug_feature_null experiments run without error at minimal scale.
# Runs locally (CPU). Each step should complete in < 5 min.
#
# Usage:
#   cd /path/to/multi-onco
#   bash experiments/03_drug_feature_null/smoke_test.sh
#
# Exit codes: 0 = all passed, non-zero = first failure.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift
    echo ""
    echo "══════════════════════════════════════════"
    echo "  SMOKE: $name"
    echo "══════════════════════════════════════════"
    if uv run python3 "$@" 2>&1; then
        echo "  ✓ PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  ✗ FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# 01 Oracle bounds (fast, CPU)
# ---------------------------------------------------------------------------
run_test "01_oracle_bounds" \
    experiments/03_drug_feature_null/01_oracle_bounds/jobs/run.py

# ---------------------------------------------------------------------------
# 02 Representation ablation: 1 fold, 2 conditions only, no sweeps
# ---------------------------------------------------------------------------
run_test "02_representation_ablation (no_drug)" \
    experiments/03_drug_feature_null/02_representation_ablation/jobs/run.py \
    --only no_drug --no-alpha-sensitivity

run_test "02_representation_ablation (morgan_fp, no sweep)" \
    experiments/03_drug_feature_null/02_representation_ablation/jobs/run.py \
    --only morgan_fp --no-alpha-sensitivity

# ---------------------------------------------------------------------------
# 02 Power analysis
# ---------------------------------------------------------------------------
run_test "02_power_analysis" \
    experiments/03_drug_feature_null/02_representation_ablation/jobs/power_analysis.py

# ---------------------------------------------------------------------------
# 04 Split robustness (fast, CPU, Ridge)
# ---------------------------------------------------------------------------
run_test "04_split_robustness" \
    experiments/03_drug_feature_null/04_split_robustness/jobs/run.py

# ---------------------------------------------------------------------------
# 05 Dataset robustness — Ridge on PRISM (CPU)
# ---------------------------------------------------------------------------
run_test "05_dataset_robustness (ridge)" \
    experiments/03_drug_feature_null/05_dataset_robustness/jobs/run_ridge.py

# ---------------------------------------------------------------------------
# 06 Objective axis: smoke mode (3 epochs, fold 0)
# ---------------------------------------------------------------------------
run_test "06_objective_axis (smoke, fold 0)" \
    experiments/03_drug_feature_null/06_objective_axis/jobs/run.py \
    --smoke --fold 0

# ---------------------------------------------------------------------------
# 07 Cross-dataset transfer (Ridge, CPU)
# ---------------------------------------------------------------------------
run_test "07_cross_dataset_transfer" \
    experiments/03_drug_feature_null/07_cross_dataset_transfer/jobs/run.py

# ---------------------------------------------------------------------------
# 03 Part A: smoke mode (3 epochs, fold 0, shard output)
# ---------------------------------------------------------------------------
run_test "03_model_robustness partA (smoke, fold 0)" \
    experiments/03_drug_feature_null/03_model_robustness/jobs/run_partA.py \
    --smoke --fold 0

# ---------------------------------------------------------------------------
# 03 Part B: smoke mode (3 epochs, fold 0, shard output)
# ---------------------------------------------------------------------------
run_test "03_model_robustness partB (smoke, fold 0)" \
    experiments/03_drug_feature_null/03_model_robustness/jobs/run_partB.py \
    --smoke --fold 0

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "══════════════════════════════════════════"
echo "  SMOKE TEST SUMMARY"
echo "══════════════════════════════════════════"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "  RESULT: ✗ FAILED ($FAIL tests failed)"
    exit 1
else
    echo "  RESULT: ✓ ALL PASSED"
    exit 0
fi
