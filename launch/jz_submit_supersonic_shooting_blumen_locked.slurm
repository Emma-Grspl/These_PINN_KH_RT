#!/bin/bash
#SBATCH --job-name=KH_shoot_blumenlock
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
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

export POINTS="${POINTS:-0.150:1.200 0.175:1.200 0.200:1.200 0.200:1.250 0.200:1.275 0.200:1.300 0.100:1.300 0.125:1.300 0.150:1.300 0.175:1.300 0.200:1.300}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_blumen_locked_reference}"
export ANCHOR_CSV="${ANCHOR_CSV:-assets/classic_supersonic/validated_modal_points/supersonic_validated_modal_points.csv}"

export MATCH_Y="${MATCH_Y:-1.0}"
export MAPPING_SCALE="${MAPPING_SCALE:-5.0}"
export MIN_Y_LIMIT="${MIN_Y_LIMIT:-10.0}"
export MAX_Y_LIMIT="${MAX_Y_LIMIT:-2000.0}"
export Y_LIMIT_FACTOR="${Y_LIMIT_FACTOR:-18.0}"
export AMP_LOWER_BOUND="${AMP_LOWER_BOUND:--30.0}"
export AMP_UPPER_BOUND="${AMP_UPPER_BOUND:-5.0}"

export CR_HALF_WINDOWS="${CR_HALF_WINDOWS:-0.004 0.008 0.015 0.03}"
export CI_HALF_WINDOWS="${CI_HALF_WINDOWS:-0.002 0.004 0.008 0.015}"
export RETRY_GROWTH="${RETRY_GROWTH:-1.35}"
export MAX_RETRIES="${MAX_RETRIES:-1}"
export MAX_ITER="${MAX_ITER:-12}"
export GRID_SIZE="${GRID_SIZE:-5}"
export ROBUST_TOP_K="${ROBUST_TOP_K:-8}"

export MAX_STAGE1="${MAX_STAGE1:-5e-2}"
export MAX_STAGE2="${MAX_STAGE2:-1e-4}"
export MAX_CI_ABS="${MAX_CI_ABS:-8e-3}"
export MAX_CI_REL="${MAX_CI_REL:-0.10}"
export MAX_CR_ABS="${MAX_CR_ABS:-3.5e-2}"
export MAX_EDGE_AMP="${MAX_EDGE_AMP:-5e-2}"
export BOX_ROBUSTNESS_FACTORS="${BOX_ROBUSTNESS_FACTORS:-1.25 1.50}"
export BOX_ROBUSTNESS_MAX_REL_L2="${BOX_ROBUSTNESS_MAX_REL_L2:-0.25}"
export BOX_ROBUSTNESS_MAX_PEAK_SHIFT="${BOX_ROBUSTNESS_MAX_PEAK_SHIFT:-1.0}"
export BOX_ROBUSTNESS_MAX_CENTER8_DELTA="${BOX_ROBUSTNESS_MAX_CENTER8_DELTA:-0.10}"
export BOX_ROBUSTNESS_MAX_EDGE_GROWTH="${BOX_ROBUSTNESS_MAX_EDGE_GROWTH:-1.25}"

echo "Blumen-locked supersonic shooting on ${SLURMD_NODENAME}"
echo "POINTS=${POINTS}"
echo "ANCHOR_CSV=${ANCHOR_CSV}"
echo "OUTPUT_STEM=${OUTPUT_STEM}"
echo "box: min=${MIN_Y_LIMIT} max=${MAX_Y_LIMIT} factor=${Y_LIMIT_FACTOR}"
echo "strict: stage1=${MAX_STAGE1} stage2=${MAX_STAGE2} ci_abs=${MAX_CI_ABS} ci_rel=${MAX_CI_REL} cr_abs=${MAX_CR_ABS}"
date

srun python3 scripts/audit_supersonic_shooting_blumen_locked.py \
  --points ${POINTS} \
  --anchor-csv "${ANCHOR_CSV}" \
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
  --robust-top-k "${ROBUST_TOP_K}" \
  --max-stage1 "${MAX_STAGE1}" \
  --max-stage2 "${MAX_STAGE2}" \
  --max-ci-abs "${MAX_CI_ABS}" \
  --max-ci-rel "${MAX_CI_REL}" \
  --max-cr-abs "${MAX_CR_ABS}" \
  --max-edge-amp "${MAX_EDGE_AMP}" \
  --box-robustness-factors ${BOX_ROBUSTNESS_FACTORS} \
  --box-robustness-max-rel-l2 "${BOX_ROBUSTNESS_MAX_REL_L2}" \
  --box-robustness-max-peak-shift "${BOX_ROBUSTNESS_MAX_PEAK_SHIFT}" \
  --box-robustness-max-center8-delta "${BOX_ROBUSTNESS_MAX_CENTER8_DELTA}" \
  --box-robustness-max-edge-growth "${BOX_ROBUSTNESS_MAX_EDGE_GROWTH}" \
  --output-stem "${OUTPUT_STEM}"
