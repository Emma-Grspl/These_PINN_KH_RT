#!/bin/bash
#SBATCH --job-name=KH_shoot_modal_surface
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

echo "Tracking modal surface supersonique sur ${SLURMD_NODENAME}"
echo "points=${POINTS_CSV:-assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv}"
echo "anchors=${ANCHOR_CSV:-assets/classic_supersonic/shooting/supersonic_reference_core_local_modal.csv}"
echo "mode_distance<=${MAX_MODE_DISTANCE:-0.35} stage1<=${MAX_STAGE1:-0.05} stage2<=${MAX_STAGE2:-0.1}"
date

srun python3 scripts/track_supersonic_shooting_modal_surface.py \
  --points-csv "${POINTS_CSV:-assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv}" \
  --anchor-csv "${ANCHOR_CSV:-assets/classic_supersonic/shooting/supersonic_reference_core_local_modal.csv}" \
  --anchor-fields-csv "${ANCHOR_FIELDS_CSV:-assets/classic_supersonic/shooting/supersonic_reference_core_local_modal_fields.csv}" \
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
  --max-mode-distance "${MAX_MODE_DISTANCE:-0.35}" \
  --max-stage1 "${MAX_STAGE1:-0.05}" \
  --max-stage2 "${MAX_STAGE2:-0.10}" \
  --max-err-ci-abs "${MAX_ERR_CI_ABS:-0.01}" \
  --max-delta-ci "${MAX_DELTA_CI:-0.025}" \
  --max-delta-cr "${MAX_DELTA_CR:-0.08}" \
  --edge-amp-threshold "${EDGE_AMP_THRESHOLD:-0.05}" \
  --output-stem "${OUTPUT_STEM:-supersonic_shooting_modal_surface_core}"
