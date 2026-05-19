#!/bin/bash
# Submit all 03_baselines stages with SLURM dependencies.
# Usage: bash launch.sh [--dry-run]
#
# Only mandatory dependency: Stage 2B reads capacity_sweep_best_configs.json
# written by Stage 2A. All other stages are independent and submit immediately.
#
# Execution order:
#   Stage 0, 1, 2A, 3  →  submit simultaneously
#   Stage 2B ×5         →  submit after Stage 2A (afterok)

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

STAGE0=$(submit "$JOBS_DIR/compute_oracles.sbatch")
echo "Submitted stage0 (oracles): job $STAGE0"

STAGE1=$(submit "$JOBS_DIR/run_ridge.sbatch")
echo "Submitted stage1 (ridge): job $STAGE1"

STAGE2A=$(submit "$JOBS_DIR/run_capacity_sweep.sbatch")
echo "Submitted stage2A (capacity_sweep): job $STAGE2A"

STAGE3=$(submit "$JOBS_DIR/run_cellblind_reg_sweep.sbatch")
echo "Submitted stage3 (cellblind_reg_sweep): job $STAGE3"

VARIANTS=(rna_only rna_mut rna_mut_cnv rna_mut_cnv_met all_5_omics)
STAGE2B_IDS=()
for VARIANT in "${VARIANTS[@]}"; do
    JID=$(submit --dependency="afterok:$STAGE2A" \
          --export="ALL,VARIANT=$VARIANT" \
          "$JOBS_DIR/run_modality_ablation.sbatch")
    STAGE2B_IDS+=("$JID")
    echo "Submitted stage2B ($VARIANT, after $STAGE2A): job $JID"
done

ALL_IDS="$STAGE0,$STAGE1,$STAGE2A,$STAGE3,$(IFS=,; echo "${STAGE2B_IDS[*]}")"
echo ""
echo "Monitor with: squeue -j $ALL_IDS"
