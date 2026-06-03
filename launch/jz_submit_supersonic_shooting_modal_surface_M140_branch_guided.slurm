#!/bin/bash
#SBATCH --job-name=KH_shoot_modal_surface_M140
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --qos=qos_cpu-t3

set -euo pipefail

module purge
module load python/3.11.5

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log
mkdir -p assets/classic_supersonic/shooting

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Preparation + relance modal surface supersonique M1.4 branch-guided sur ${SLURMD_NODENAME}"
echo "branch-guided-summary=${BRANCH_GUIDED_SUMMARY:-assets/classic_supersonic/shooting/supersonic_shooting_point_batch_M140_branch_guided_summary.csv}"
echo "branch-guided-fields=${BRANCH_GUIDED_FIELDS:-assets/classic_supersonic/shooting/supersonic_shooting_point_batch_M140_branch_guided_fields.csv}"
echo "base-points=${POINTS_REF:-assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv}"
echo "base-anchors=${ANCHOR_REF:-assets/classic_supersonic/shooting/supersonic_reference_core_local_modal.csv}"
echo "cross-mach-alpha-tolerance=${CROSS_MACH_ALPHA_TOLERANCE:-0.02}"
date

PREP_POINTS="${PREP_POINTS:-assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral_M140_branch_guided.csv}"
PREP_ANCHORS="${PREP_ANCHORS:-assets/classic_supersonic/shooting/supersonic_reference_core_local_modal_M140_branch_guided.csv}"
PREP_ANCHOR_FIELDS="${PREP_ANCHOR_FIELDS:-assets/classic_supersonic/shooting/supersonic_reference_core_local_modal_fields_M140_branch_guided.csv}"
PREP_REPORT="${PREP_REPORT:-assets/classic_supersonic/shooting/supersonic_reference_core_local_M140_branch_guided_report.csv}"

python3 scripts/prepare_supersonic_modal_surface_branch_guided_inputs.py \
  --branch-guided-summary "${BRANCH_GUIDED_SUMMARY:-assets/classic_supersonic/shooting/supersonic_shooting_point_batch_M140_branch_guided_summary.csv}" \
  --branch-guided-fields "${BRANCH_GUIDED_FIELDS:-assets/classic_supersonic/shooting/supersonic_shooting_point_batch_M140_branch_guided_fields.csv}" \
  --points-ref "${POINTS_REF:-assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv}" \
  --anchor-ref "${ANCHOR_REF:-assets/classic_supersonic/shooting/supersonic_reference_core_local_modal.csv}" \
  --anchor-fields-ref "${ANCHOR_FIELDS_REF:-assets/classic_supersonic/shooting/supersonic_reference_core_local_modal_fields.csv}" \
  --output-points "${PREP_POINTS}" \
  --output-anchors "${PREP_ANCHORS}" \
  --output-anchor-fields "${PREP_ANCHOR_FIELDS}" \
  --report-path "${PREP_REPORT}"

srun python3 scripts/track_supersonic_shooting_modal_surface.py \
  --points-csv "${PREP_POINTS}" \
  --anchor-csv "${PREP_ANCHORS}" \
  --anchor-fields-csv "${PREP_ANCHOR_FIELDS}" \
  --match-y "${MATCH_Y:-1.0}" \
  --use-mapping \
  --mapping-scale "${MAPPING_SCALE:-5.0}" \
  --min-y-limit "${MIN_Y_LIMIT:-10.0}" \
  --max-y-limit "${MAX_Y_LIMIT:-500.0}" \
  --y-limit-factor "${Y_LIMIT_FACTOR:-6.0}" \
  --amp-lower-bound "${AMP_LOWER_BOUND:--30.0}" \
  --amp-upper-bound "${AMP_UPPER_BOUND:-5.0}" \
  --cr-half-windows ${CR_HALF_WINDOWS:-0.015 0.03} \
  --ci-half-windows ${CI_HALF_WINDOWS:-0.008 0.015} \
  --retry-growth "${RETRY_GROWTH:-1.60}" \
  --max-retries "${MAX_RETRIES:-2}" \
  --max-iter "${MAX_ITER:-10}" \
  --grid-size "${GRID_SIZE:-4}" \
  --ci-weight "${CI_WEIGHT:-4.0}" \
  --cr-weight "${CR_WEIGHT:-0.35}" \
  --continuity-weight "${CONTINUITY_WEIGHT:-1.0}" \
  --mode-distance-weight "${MODE_DISTANCE_WEIGHT:-0.60}" \
  --mode-mask-ratio "${MODE_MASK_RATIO:-0.15}" \
  --mode-common-points "${MODE_COMMON_POINTS:-801}" \
  --cross-mach-alpha-tolerance "${CROSS_MACH_ALPHA_TOLERANCE:-0.02}" \
  --max-mode-distance "${MAX_MODE_DISTANCE:-0.35}" \
  --max-stage1 "${MAX_STAGE1:-0.05}" \
  --max-stage2 "${MAX_STAGE2:-0.10}" \
  --max-err-ci-abs "${MAX_ERR_CI_ABS:-0.01}" \
  --max-delta-ci "${MAX_DELTA_CI:-0.025}" \
  --max-delta-cr "${MAX_DELTA_CR:-0.08}" \
  --edge-amp-threshold "${EDGE_AMP_THRESHOLD:-0.05}" \
  --output-stem "${OUTPUT_STEM:-supersonic_shooting_modal_surface_core_M140_branch_guided}"
