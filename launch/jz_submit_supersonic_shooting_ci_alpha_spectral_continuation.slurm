#!/bin/bash
#SBATCH --job-name=KH_shoot_ci_spec
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
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
export LINE_SPECS="${LINE_SPECS:-1.20:0.20:0.15,0.175,0.20,0.225,0.25,0.275 1.30:0.175:0.10,0.125,0.15,0.175,0.20,0.225 1.50:0.125:0.125,0.15,0.175,0.20,0.225,0.25}"
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
export MAX_ERR_CI_ABS="${MAX_ERR_CI_ABS:-0.01}"
export MAX_DELTA_CI="${MAX_DELTA_CI:-0.025}"
export MAX_DELTA_CR="${MAX_DELTA_CR:-0.08}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_ci_alpha_spectral_continuation_core}"

echo "Audit supersonique shooting c_i(alpha) en continuation spectrale sur ${SLURMD_NODENAME}"
echo "line-specs=${LINE_SPECS}"
echo "workers=${WORKERS}"
echo "box: min=${MIN_Y_LIMIT} max=${MAX_Y_LIMIT} factor=${Y_LIMIT_FACTOR} amp=[${AMP_LOWER_BOUND},${AMP_UPPER_BOUND}]"
echo "guards: stage1<=${MAX_STAGE1} err_ci<=${MAX_ERR_CI_ABS} delta_ci<=${MAX_DELTA_CI} delta_cr<=${MAX_DELTA_CR}"
date

srun python3 scripts/audit_supersonic_shooting_ci_alpha_continuation.py \
  --line-specs ${LINE_SPECS} \
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
  --acceptance-mode spectral \
  --edge-amp-threshold "${EDGE_AMP_THRESHOLD}" \
  --max-stage1 "${MAX_STAGE1}" \
  --max-err-ci-abs "${MAX_ERR_CI_ABS}" \
  --max-delta-ci "${MAX_DELTA_CI}" \
  --max-delta-cr "${MAX_DELTA_CR}" \
  --output-stem "${OUTPUT_STEM}"
