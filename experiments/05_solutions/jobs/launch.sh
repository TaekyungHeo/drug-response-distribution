#!/bin/bash
# Submit all 05_solutions experiments to SLURM with dependency chain.
# Usage: bash experiments/05_solutions/jobs/launch.sh [--dry-run]
set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

submit() {
    if $DRY_RUN; then
        echo "[dry-run] sbatch $*" >&2
        echo "99999"
    else
        sbatch --parsable "$@"
    fi
}

echo "=== 05_solutions SLURM launch ==="
echo ""

echo "Phase 1: Independent experiments"
J1=$(submit "$BASE/01_diagnosis/01_moa_performance/jobs/sbatch.sh")
J2=$(submit "$BASE/01_diagnosis/02_moa_ceiling/jobs/sbatch.sh")
J3=$(submit "$BASE/04_external_signatures/01_lincs/jobs/sbatch.sh")
J4=$(submit "$BASE/04_external_signatures/02_lincs_prediction/jobs/sbatch.sh")
echo "  01_moa_performance: $J1"
echo "  02_moa_ceiling:     $J2"
echo "  01_lincs:           $J3"
echo "  02_lincs_pred:      $J4"
echo ""

echo "Phase 2: Training distribution (after diagnosis)"
J5=$(submit --dependency=afterok:$J1 "$BASE/02_training_distribution/01_within_moa/jobs/sbatch.sh")
J6=$(submit --dependency=afterok:$J1 "$BASE/02_training_distribution/02_moa_weighted/jobs/sbatch.sh")
J7=$(submit --dependency=afterok:$J1 "$BASE/02_training_distribution/03_onehot_control/jobs/sbatch.sh")
echo "  01_within_moa:      $J5 (after $J1)"
echo "  02_moa_weighted:    $J6 (after $J1)"
echo "  03_onehot_control:  $J7 (after $J1)"
echo ""

echo "Phase 3: Few-shot (independent)"
J8=$(submit "$BASE/03_few_shot/01_response_matching/jobs/sbatch.sh")
J9=$(submit --dependency=afterok:$J8 "$BASE/03_few_shot/02_active_selection/jobs/sbatch.sh")
echo "  01_response_match:  $J8"
echo "  02_active_select:   $J9 (after $J8)"
echo ""

echo "Phase 4: Combinations (after phase 2+3)"
J10=$(submit --dependency=afterok:$J5:$J8 "$BASE/05_combinations/01_moa_x_kshot/jobs/sbatch.sh")
J11=$(submit --dependency=afterok:$J3:$J5 "$BASE/05_combinations/02_lincs_x_moa/jobs/sbatch.sh")
echo "  01_moa_x_kshot:     $J10 (after $J5,$J8)"
echo "  02_lincs_x_moa:     $J11 (after $J3,$J5)"
echo ""

ALL_IDS="$J1,$J2,$J3,$J4,$J5,$J6,$J7,$J8,$J9,$J10,$J11"
echo "All 11 jobs submitted."
echo "Monitor: squeue -j $ALL_IDS"
