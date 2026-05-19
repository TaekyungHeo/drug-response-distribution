#!/bin/bash
# Submit all three steps with SLURM dependencies.
# Usage: bash launch.sh [--dry-run]
#
# Step 1 (array 0-4) → Step 2 (array 0-4) → Step 3 (single job)

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

submit() {
    if $DRY_RUN; then
        echo "[dry-run] sbatch $*" >&2
        echo "99999"
    else
        sbatch --parsable "$@"
    fi
}

mkdir -p "$JOBS_DIR/../logs"

STEP1=$(submit "$JOBS_DIR/step1_ridge.sbatch")
echo "Submitted step1 (array): job $STEP1"

STEP2=$(submit --dependency="afterok:$STEP1" "$JOBS_DIR/step2_bootstrap.sbatch")
echo "Submitted step2 (array, after $STEP1): job $STEP2"

STEP3=$(submit --dependency="afterok:$STEP2" "$JOBS_DIR/step3_aggregate.sbatch")
echo "Submitted step3 (after $STEP2): job $STEP3"

echo ""
echo "Monitor with: squeue -j $STEP1,$STEP2,$STEP3"
