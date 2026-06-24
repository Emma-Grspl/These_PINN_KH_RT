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
echo "OUTPUT_STEM=${OUTPUT_STEM}"
echo "MAX_CANDIDATES_PER_POINT=${MAX_CANDIDATES_PER_POINT}"
echo "CANDIDATE_SOURCE=${CANDIDATE_SOURCE}"
echo "box: required=${BOX_REQUIRED} min=${BOX_MIN} max=${BOX_MAX} factor=${BOX_FACTOR}"
echo "strict: stage1=${STRICT_STAGE1} stage2=${STRICT_STAGE2} ci_abs=${STRICT_CI_ABS} ci_rel=${STRICT_CI_REL} cr_abs=${STRICT_CR_ABS}"
date

srun python3 scripts/audit_supersonic_shooting_blumen_locked_multicandidate.py \
  --points ${POINTS} \
  --anchor-csv "${ANCHOR_CSV}" \
  --output-stem "${OUTPUT_STEM}" \
  --candidate-source "${CANDIDATE_SOURCE}" \
  --max-candidates-per-point "${MAX_CANDIDATES_PER_POINT}" \
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
