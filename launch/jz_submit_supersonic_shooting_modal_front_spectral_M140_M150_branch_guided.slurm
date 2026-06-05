#!/bin/bash
#SBATCH --job-name=KH_shoot_front_M140_M150
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

export WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK}}"
export POINTS_CSV="${POINTS_CSV:-assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral_M140_branch_guided.csv}"
export MODAL_SUMMARY_CSV="${MODAL_SUMMARY_CSV:-assets/classic_supersonic/shooting/supersonic_shooting_modal_surface_core_M140_branch_guided_relaxed_summary.csv}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_modal_front_spectral_dense_M140_M150_branch_guided}"

# Front local cible :
# - M=1.4 : prolonger 0.1875 -> 0.19375 -> 0.2000
# - M=1.5 : prolonger 0.1625 -> 0.16875 -> 0.1750 -> 0.18125
export MANUAL_LINE_SPECS="${MANUAL_LINE_SPECS:-1.40:0.187500:0.175000,0.181250,0.187500,0.190625,0.193750,0.196875,0.200000 1.50:0.162500:0.150000,0.156250,0.162500,0.165625,0.168750,0.171875,0.175000,0.178125,0.181250}"
export MANUAL_ONLY="${MANUAL_ONLY:-1}"
export DRY_RUN="${DRY_RUN:-0}"

export MATCH_Y="${MATCH_Y:-1.0}"
export MAPPING_SCALE="${MAPPING_SCALE:-5.0}"
export MIN_Y_LIMIT="${MIN_Y_LIMIT:-10.0}"
export MAX_Y_LIMIT="${MAX_Y_LIMIT:-500.0}"
export Y_LIMIT_FACTOR="${Y_LIMIT_FACTOR:-6.0}"
export AMP_LOWER_BOUND="${AMP_LOWER_BOUND:--30.0}"
export AMP_UPPER_BOUND="${AMP_UPPER_BOUND:-5.0}"
export CR_HALF_WINDOWS="${CR_HALF_WINDOWS:-0.015 0.03}"
export CI_HALF_WINDOWS="${CI_HALF_WINDOWS:-0.008 0.015}"
export RETRY_GROWTH="${RETRY_GROWTH:-1.60}"
export MAX_RETRIES="${MAX_RETRIES:-2}"
export MAX_ITER="${MAX_ITER:-10}"
export GRID_SIZE="${GRID_SIZE:-4}"
export CI_WEIGHT="${CI_WEIGHT:-4.0}"
export CR_WEIGHT="${CR_WEIGHT:-0.35}"
export CONTINUITY_WEIGHT="${CONTINUITY_WEIGHT:-1.0}"
export EDGE_AMP_THRESHOLD="${EDGE_AMP_THRESHOLD:-0.05}"
export MAX_STAGE1="${MAX_STAGE1:-0.05}"
export MAX_ERR_CI_ABS="${MAX_ERR_CI_ABS:-0.02}"
export MAX_DELTA_CI="${MAX_DELTA_CI:-0.025}"
export MAX_DELTA_CR="${MAX_DELTA_CR:-0.08}"

echo "Densification locale front supersonique M1.4/M1.5 sur ${SLURMD_NODENAME}"
echo "points=${POINTS_CSV}"
echo "modal-summary=${MODAL_SUMMARY_CSV}"
echo "workers=${WORKERS}"
echo "manual-only=${MANUAL_ONLY}"
echo "manual-line-specs=${MANUAL_LINE_SPECS}"
echo "box: min=${MIN_Y_LIMIT} max=${MAX_Y_LIMIT} factor=${Y_LIMIT_FACTOR} amp=[${AMP_LOWER_BOUND},${AMP_UPPER_BOUND}]"
echo "guards: stage1<=${MAX_STAGE1} err_ci<=${MAX_ERR_CI_ABS} delta_ci<=${MAX_DELTA_CI} delta_cr<=${MAX_DELTA_CR}"
date

EXTRA_ARGS=()
if [ -n "${MANUAL_LINE_SPECS}" ]; then
  EXTRA_ARGS+=(--manual-line-specs ${MANUAL_LINE_SPECS})
fi
if [ "${MANUAL_ONLY}" = "1" ]; then
  EXTRA_ARGS+=(--manual-only)
fi
if [ "${DRY_RUN}" = "1" ]; then
  EXTRA_ARGS+=(--dry-run)
fi

srun python3 scripts/densify_supersonic_modal_front_spectral.py \
  --points-csv "${POINTS_CSV}" \
  --modal-summary-csv "${MODAL_SUMMARY_CSV}" \
  --workers "${WORKERS}" \
  --match-y "${MATCH_Y}" \
  --use-mapping \
  --mapping-scale "${MAPPING_SCALE}" \
  --min-y-limit "${MIN_Y_LIMIT}" \
  --max-y-limit "${MAX_Y_LIMIT}" \
  --y-limit-factor "${Y_LIMIT_FACTOR}" \
  --amp-lower-bound "${AMP_LOWER_BOUND}" \
  --amp-upper-bound "${AMP_UPPER_BOUND}" \
  --cr-half-windows ${CR_HALF_WINDOWS} \
  --ci-half-windows ${CI_HALF_WINDOWS} \
  --retry-growth "${RETRY_GROWTH}" \
  --max-retries "${MAX_RETRIES}" \
  --max-iter "${MAX_ITER}" \
  --grid-size "${GRID_SIZE}" \
  --ci-weight "${CI_WEIGHT}" \
  --cr-weight "${CR_WEIGHT}" \
  --continuity-weight "${CONTINUITY_WEIGHT}" \
  --edge-amp-threshold "${EDGE_AMP_THRESHOLD}" \
  --max-stage1 "${MAX_STAGE1}" \
  --max-err-ci-abs "${MAX_ERR_CI_ABS}" \
  --max-delta-ci "${MAX_DELTA_CI}" \
  --max-delta-cr "${MAX_DELTA_CR}" \
  --output-stem "${OUTPUT_STEM}" \
  "${EXTRA_ARGS[@]}"
