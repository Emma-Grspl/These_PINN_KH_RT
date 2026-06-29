#!/bin/bash
#SBATCH --job-name=KH_shoot_multi
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_cpu-t3

set -euo pipefail

module purge
module load python/3.11.5

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log
mkdir -p assets/classic_supersonic/multicandidate_audits

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export POINTS="${POINTS:-0.200:1.800}"
export ANCHOR_CSV="${ANCHOR_CSV:-assets/classic_supersonic/validated_modal_points/supersonic_validated_modal_points.csv}"
export MODAL_REFERENCE_CSV="${MODAL_REFERENCE_CSV:-${ANCHOR_CSV}}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_blumen_locked_multicandidate_reference}"
export MAX_CANDIDATES_PER_POINT="${MAX_CANDIDATES_PER_POINT:-12}"
export BOX_REQUIRED="${BOX_REQUIRED:-1}"
export BOX_MIN="${BOX_MIN:-10.0}"
export BOX_MAX="${BOX_MAX:-1200.0}"
export BOX_FACTOR="${BOX_FACTOR:-10.0}"
export STRICT_STAGE1="${STRICT_STAGE1:-5e-2}"
export STRICT_STAGE2="${STRICT_STAGE2:-1e-4}"
export STRICT_CI_ABS="${STRICT_CI_ABS:-8e-3}"
export STRICT_CI_REL="${STRICT_CI_REL:-0.10}"
export STRICT_CR_ABS="${STRICT_CR_ABS:-3.5e-2}"
export STRICT_BOX_MAX_REL_L2="${STRICT_BOX_MAX_REL_L2:-0.15}"
export STRICT_PEAK_SHIFT="${STRICT_PEAK_SHIFT:-0.75}"
export CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-auto}"
export MANUAL_SEED_GRID="${MANUAL_SEED_GRID:-}"
export MAX_RETRIES="${MAX_RETRIES:-}"
export MAX_ITER="${MAX_ITER:-}"
export GRID_SIZE="${GRID_SIZE:-}"
export CR_HALF_WINDOWS="${CR_HALF_WINDOWS:-}"
export CI_HALF_WINDOWS="${CI_HALF_WINDOWS:-}"
export EXISTING_MACH_WINDOW="${EXISTING_MACH_WINDOW:-}"
export DRY_RUN_CANDIDATES="${DRY_RUN_CANDIDATES:-0}"
export STOP_AFTER_ACCEPTED="${STOP_AFTER_ACCEPTED:-1}"
export FLUSH_EVERY_CANDIDATE="${FLUSH_EVERY_CANDIDATE:-1}"

BOX_FLAG="--box-required"
if [ "${BOX_REQUIRED}" = "0" ]; then
  BOX_FLAG="--no-box-required"
fi

DRY_FLAG=""
if [ "${DRY_RUN_CANDIDATES}" = "1" ]; then
  DRY_FLAG="--dry-run-candidates"
fi

STOP_FLAG="--stop-after-accepted"
if [ "${STOP_AFTER_ACCEPTED}" = "0" ]; then
  STOP_FLAG="--no-stop-after-accepted"
fi

FLUSH_FLAG="--flush-every-candidate"
if [ "${FLUSH_EVERY_CANDIDATE}" = "0" ]; then
  FLUSH_FLAG="--no-flush-every-candidate"
fi

echo "Supersonic multicandidate audit on ${SLURMD_NODENAME}"
echo "POINTS=${POINTS}"
echo "ANCHOR_CSV=${ANCHOR_CSV}"
echo "MODAL_REFERENCE_CSV=${MODAL_REFERENCE_CSV}"
echo "OUTPUT_STEM=${OUTPUT_STEM}"
echo "MAX_CANDIDATES_PER_POINT=${MAX_CANDIDATES_PER_POINT}"
echo "CANDIDATE_SOURCE=${CANDIDATE_SOURCE}"
echo "MANUAL_SEED_GRID=${MANUAL_SEED_GRID}"
echo "MAX_RETRIES=${MAX_RETRIES}"
echo "MAX_ITER=${MAX_ITER}"
echo "GRID_SIZE=${GRID_SIZE}"
echo "CR_HALF_WINDOWS=${CR_HALF_WINDOWS}"
echo "CI_HALF_WINDOWS=${CI_HALF_WINDOWS}"
echo "EXISTING_MACH_WINDOW=${EXISTING_MACH_WINDOW}"
echo "box: required=${BOX_REQUIRED} min=${BOX_MIN} max=${BOX_MAX} factor=${BOX_FACTOR}"
echo "strict: stage1=${STRICT_STAGE1} stage2=${STRICT_STAGE2} ci_abs=${STRICT_CI_ABS} ci_rel=${STRICT_CI_REL} cr_abs=${STRICT_CR_ABS}"
date

EXTRA_ARGS=()
if [[ -n "${MANUAL_SEED_GRID}" ]]; then EXTRA_ARGS+=(--manual-seed-grid "${MANUAL_SEED_GRID}"); fi
if [[ -n "${MAX_RETRIES}" ]]; then EXTRA_ARGS+=(--max-retries "${MAX_RETRIES}"); fi
if [[ -n "${MAX_ITER}" ]]; then EXTRA_ARGS+=(--max-iter "${MAX_ITER}"); fi
if [[ -n "${GRID_SIZE}" ]]; then EXTRA_ARGS+=(--grid-size "${GRID_SIZE}"); fi
if [[ -n "${EXISTING_MACH_WINDOW}" ]]; then EXTRA_ARGS+=(--existing-mach-window "${EXISTING_MACH_WINDOW}"); fi
if [[ -n "${CR_HALF_WINDOWS}" ]]; then
  read -r -a CR_HALF_WINDOWS_ARRAY <<< "${CR_HALF_WINDOWS}"
  EXTRA_ARGS+=(--cr-half-windows "${CR_HALF_WINDOWS_ARRAY[@]}")
fi
if [[ -n "${CI_HALF_WINDOWS}" ]]; then
  read -r -a CI_HALF_WINDOWS_ARRAY <<< "${CI_HALF_WINDOWS}"
  EXTRA_ARGS+=(--ci-half-windows "${CI_HALF_WINDOWS_ARRAY[@]}")
fi

srun python3 scripts/audit_supersonic_shooting_blumen_locked_multicandidate.py \
  --points ${POINTS} \
  --anchor-csv "${ANCHOR_CSV}" \
  --modal-reference-csv "${MODAL_REFERENCE_CSV}" \
  --output-stem "${OUTPUT_STEM}" \
  --candidate-source "${CANDIDATE_SOURCE}" \
  --max-candidates-per-point "${MAX_CANDIDATES_PER_POINT}" \
  "${EXTRA_ARGS[@]}" \
  ${BOX_FLAG} \
  --box-min "${BOX_MIN}" \
  --box-max "${BOX_MAX}" \
  --box-factor "${BOX_FACTOR}" \
  --strict-stage1 "${STRICT_STAGE1}" \
  --strict-stage2 "${STRICT_STAGE2}" \
  --strict-ci-abs "${STRICT_CI_ABS}" \
  --strict-ci-rel "${STRICT_CI_REL}" \
  --strict-cr-abs "${STRICT_CR_ABS}" \
  --strict-box-max-rel-l2 "${STRICT_BOX_MAX_REL_L2}" \
  --strict-peak-shift "${STRICT_PEAK_SHIFT}" \
  ${STOP_FLAG} \
  ${FLUSH_FLAG} \
  ${DRY_FLAG}
