#!/bin/bash
#SBATCH --job-name=KH_shoot_M140_batch
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
export MACH="${MACH:-1.40}"
export ALPHAS="${ALPHAS:-0.125000,0.137500,0.150000,0.162500,0.168750,0.175000,0.181250,0.187500,0.193750,0.200000}"
export MATCH_Y="${MATCH_Y:-1.0}"
export MAPPING_SCALE="${MAPPING_SCALE:-5.0}"
export MIN_Y_LIMIT="${MIN_Y_LIMIT:-10.0}"
export MAX_Y_LIMIT="${MAX_Y_LIMIT:-500.0}"
export Y_LIMIT_FACTOR="${Y_LIMIT_FACTOR:-6.0}"
export AMP_LOWER_BOUND="${AMP_LOWER_BOUND:--30.0}"
export AMP_UPPER_BOUND="${AMP_UPPER_BOUND:-5.0}"
export CR_HALF_WINDOWS="${CR_HALF_WINDOWS:-0.015 0.03 0.06 0.10}"
export CI_HALF_WINDOWS="${CI_HALF_WINDOWS:-0.008 0.015 0.03 0.06}"
export RETRY_GROWTH="${RETRY_GROWTH:-1.75}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export MAX_ITER="${MAX_ITER:-10}"
export GRID_SIZE="${GRID_SIZE:-4}"
export CI_WEIGHT="${CI_WEIGHT:-4.0}"
export CR_WEIGHT="${CR_WEIGHT:-0.35}"
export CONTINUITY_WEIGHT="${CONTINUITY_WEIGHT:-0.20}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_point_batch_M140}"

echo "Audit pointwise shooting supersonique M=1.4 sur ${SLURMD_NODENAME}"
echo "mach=${MACH}"
echo "alphas=${ALPHAS}"
echo "workers=${WORKERS}"
echo "box: min=${MIN_Y_LIMIT} max=${MAX_Y_LIMIT} factor=${Y_LIMIT_FACTOR} amp=[${AMP_LOWER_BOUND},${AMP_UPPER_BOUND}]"
date

srun python3 scripts/audit_supersonic_shooting_point_batch.py \
  --mach "${MACH}" \
  --alphas "${ALPHAS}" \
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
  --output-stem "${OUTPUT_STEM}"
