#!/bin/bash
#SBATCH --job-name=KH_shoot_ref_pkg
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_cpu-t3

set -euo pipefail

module purge
module load python/3.11.5

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log
mkdir -p assets/blumen_gep

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export MACH_MIN="${MACH_MIN:-1.0}"
export MACH_MAX="${MACH_MAX:-2.0}"
export ALPHA_MIN="${ALPHA_MIN:-0.02}"
export ALPHA_MAX="${ALPHA_MAX:-0.50}"
export NUM_MACH="${NUM_MACH:-21}"
export NUM_ALPHA="${NUM_ALPHA:-21}"
export MATCH_Y="${MATCH_Y:-1.0}"
export MAPPING_SCALE="${MAPPING_SCALE:-5.0}"
export MIN_Y_LIMIT="${MIN_Y_LIMIT:-10.0}"
export MAX_Y_LIMIT="${MAX_Y_LIMIT:-500.0}"
export Y_LIMIT_FACTOR="${Y_LIMIT_FACTOR:-6.0}"
export AMP_LOWER_BOUND="${AMP_LOWER_BOUND:--30.0}"
export AMP_UPPER_BOUND="${AMP_UPPER_BOUND:-5.0}"
export CR_HALF_WINDOWS="${CR_HALF_WINDOWS:-0.015 0.03 0.06 0.10}"
export CI_HALF_WINDOWS="${CI_HALF_WINDOWS:-0.008 0.015 0.03}"
export RETRY_GROWTH="${RETRY_GROWTH:-1.75}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export MAX_ITER="${MAX_ITER:-10}"
export GRID_SIZE="${GRID_SIZE:-4}"
export CI_WEIGHT="${CI_WEIGHT:-4.0}"
export CR_WEIGHT="${CR_WEIGHT:-0.35}"
export CONTINUITY_WEIGHT="${CONTINUITY_WEIGHT:-0.20}"
export MODE_POINTS="${MODE_POINTS:-0.18:1.33 0.20:1.20 0.20:1.30}"
export MODE_TOP_K="${MODE_TOP_K:-2}"
export MODE_SEED_CR_COUNT="${MODE_SEED_CR_COUNT:-6}"
export MODE_SEED_CI_COUNT="${MODE_SEED_CI_COUNT:-6}"
export MODE_CR_MIN="${MODE_CR_MIN:-0.0}"
export MODE_CR_MAX="${MODE_CR_MAX:-0.60}"
export MODE_CI_MIN="${MODE_CI_MIN:-0.01}"
export MODE_CI_MAX="${MODE_CI_MAX:-0.12}"
export MODE_CR_HALF_WINDOWS="${MODE_CR_HALF_WINDOWS:-0.03 0.06}"
export MODE_CI_HALF_WINDOWS="${MODE_CI_HALF_WINDOWS:-0.015 0.03}"
export MODE_DEDUP_CR_TOL="${MODE_DEDUP_CR_TOL:-0.005}"
export MODE_DEDUP_CI_TOL="${MODE_DEDUP_CI_TOL:-0.005}"
export VISIBLE_THRESHOLD_RATIO="${VISIBLE_THRESHOLD_RATIO:-0.02}"
export VISIBLE_MIN_HALF_WIDTH="${VISIBLE_MIN_HALF_WIDTH:-8.0}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_reference_package}"

echo "Paquet de reference shooting supersonique sur ${SLURMD_NODENAME}"
echo "grid Mach: ${MACH_MIN} -> ${MACH_MAX} (${NUM_MACH} points)"
echo "grid alpha: ${ALPHA_MIN} -> ${ALPHA_MAX} (${NUM_ALPHA} points)"
echo "box: min=${MIN_Y_LIMIT} max=${MAX_Y_LIMIT} factor=${Y_LIMIT_FACTOR} amp=[${AMP_LOWER_BOUND},${AMP_UPPER_BOUND}]"
echo "mode points: ${MODE_POINTS}"
echo "output stem: ${OUTPUT_STEM}"
date

srun python3 scripts/build_supersonic_shooting_reference_package.py \
  --mach-min "${MACH_MIN}" \
  --mach-max "${MACH_MAX}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --num-mach "${NUM_MACH}" \
  --num-alpha "${NUM_ALPHA}" \
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
  --mode-points ${MODE_POINTS} \
  --mode-top-k "${MODE_TOP_K}" \
  --mode-seed-cr-count "${MODE_SEED_CR_COUNT}" \
  --mode-seed-ci-count "${MODE_SEED_CI_COUNT}" \
  --mode-cr-min "${MODE_CR_MIN}" \
  --mode-cr-max "${MODE_CR_MAX}" \
  --mode-ci-min "${MODE_CI_MIN}" \
  --mode-ci-max "${MODE_CI_MAX}" \
  --mode-cr-half-windows ${MODE_CR_HALF_WINDOWS} \
  --mode-ci-half-windows ${MODE_CI_HALF_WINDOWS} \
  --mode-dedup-cr-tol "${MODE_DEDUP_CR_TOL}" \
  --mode-dedup-ci-tol "${MODE_DEDUP_CI_TOL}" \
  --visible-threshold-ratio "${VISIBLE_THRESHOLD_RATIO}" \
  --visible-min-half-width "${VISIBLE_MIN_HALF_WIDTH}" \
  --output-stem "${OUTPUT_STEM}"
