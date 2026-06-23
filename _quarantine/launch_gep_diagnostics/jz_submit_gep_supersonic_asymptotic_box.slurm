#!/bin/bash
#SBATCH --job-name=KH_GEP_sup_asymbox
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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

export ALPHA="${ALPHA:-0.20}"
export MACH="${MACH:-1.20}"
export N_POINTS="${N_POINTS:-401}"
export AUTO_N_POINTS="${AUTO_N_POINTS:-1}"
export MAX_N_POINTS="${MAX_N_POINTS:-1201}"
export MAPPING_L="${MAPPING_L:-3.0}"
export XI_MAX="${XI_MAX:-1.0}"
export TOP_K="${TOP_K:-5}"
export POSITIVE_CR_ONLY="${POSITIVE_CR_ONLY:-1}"
export DECAY_BOX_FACTOR="${DECAY_BOX_FACTOR:-4.0}"
export OSCILLATION_BOX_FACTOR="${OSCILLATION_BOX_FACTOR:-4.0}"
export MAX_WAVELENGTH_TO_DECAY_RATIO_FOR_Q_BOX="${MAX_WAVELENGTH_TO_DECAY_RATIO_FOR_Q_BOX:-8.0}"
export MIN_HALF_BOX="${MIN_HALF_BOX:-8.0}"
export MAX_HALF_BOX="${MAX_HALF_BOX:-120.0}"
export FORCE_SYMMETRIC_HALF_BOX="${FORCE_SYMMETRIC_HALF_BOX:-0}"
export Q_FLOOR="${Q_FLOOR:-1e-4}"
export CORE_HALF_WIDTH="${CORE_HALF_WIDTH:-1.0}"
export MIN_POINTS_CORE="${MIN_POINTS_CORE:-120}"
export CENTER_HALF_WIDTH="${CENTER_HALF_WIDTH:-4.0}"
export MIN_POINTS_CENTER="${MIN_POINTS_CENTER:-300}"
export TAIL_POINTS_PER_WAVELENGTH="${TAIL_POINTS_PER_WAVELENGTH:-24.0}"
export TAIL_FRACTION_FOR_SPACING="${TAIL_FRACTION_FOR_SPACING:-0.15}"
export SHOOTING_CR_MAX="${SHOOTING_CR_MAX:-0.70}"
export SHOOTING_CI_MAX="${SHOOTING_CI_MAX:-0.12}"
export SHOOTING_MAX_ITER="${SHOOTING_MAX_ITER:-12}"
export SHOOTING_GRID_SIZE="${SHOOTING_GRID_SIZE:-4}"
export TARGET_CR="${TARGET_CR:-}"
export TARGET_CI="${TARGET_CI:-}"
export U_ZOOM_HALF_WIDTH="${U_ZOOM_HALF_WIDTH:-2.0}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_gep_asymptotic_box}"

echo "Audit GEP supersonique asymptotic-box sur ${SLURMD_NODENAME}"
echo "alpha=${ALPHA} Mach=${MACH}"
echo "N_POINTS=${N_POINTS} AUTO_N_POINTS=${AUTO_N_POINTS} MAX_N_POINTS=${MAX_N_POINTS}"
echo "mapping L=${MAPPING_L} xi_max=${XI_MAX}"
echo "box factors: decay=${DECAY_BOX_FACTOR} oscillation=${OSCILLATION_BOX_FACTOR}"
echo "q-box ratio cutoff=${MAX_WAVELENGTH_TO_DECAY_RATIO_FOR_Q_BOX}"
echo "force symmetric half box=${FORCE_SYMMETRIC_HALF_BOX}"
echo "target override: cr=${TARGET_CR:-auto} ci=${TARGET_CI:-auto}"
echo "resolution targets: core<=${CORE_HALF_WIDTH} -> ${MIN_POINTS_CORE} pts | center<=${CENTER_HALF_WIDTH} -> ${MIN_POINTS_CENTER} pts | tail ppw=${TAIL_POINTS_PER_WAVELENGTH}"
echo "output stem=${OUTPUT_STEM}"
date

ARGS=(
  --alpha "${ALPHA}"
  --mach "${MACH}"
  --n-points "${N_POINTS}"
  --max-n-points "${MAX_N_POINTS}"
  --mapping-L "${MAPPING_L}"
  --xi-max "${XI_MAX}"
  --top-k "${TOP_K}"
  --decay-box-factor "${DECAY_BOX_FACTOR}"
  --oscillation-box-factor "${OSCILLATION_BOX_FACTOR}"
  --max-wavelength-to-decay-ratio-for-q-box "${MAX_WAVELENGTH_TO_DECAY_RATIO_FOR_Q_BOX}"
  --min-half-box "${MIN_HALF_BOX}"
  --max-half-box "${MAX_HALF_BOX}"
  --q-floor "${Q_FLOOR}"
  --core-half-width "${CORE_HALF_WIDTH}"
  --min-points-core "${MIN_POINTS_CORE}"
  --center-half-width "${CENTER_HALF_WIDTH}"
  --min-points-center "${MIN_POINTS_CENTER}"
  --tail-points-per-wavelength "${TAIL_POINTS_PER_WAVELENGTH}"
  --tail-fraction-for-spacing "${TAIL_FRACTION_FOR_SPACING}"
  --shooting-cr-max "${SHOOTING_CR_MAX}"
  --shooting-ci-max "${SHOOTING_CI_MAX}"
  --shooting-max-iter "${SHOOTING_MAX_ITER}"
  --shooting-grid-size "${SHOOTING_GRID_SIZE}"
  --u-zoom-half-width "${U_ZOOM_HALF_WIDTH}"
  --output-stem "${OUTPUT_STEM}"
)

if [ -n "${TARGET_CR}" ] && [ -n "${TARGET_CI}" ]; then
  ARGS+=(--target-cr "${TARGET_CR}" --target-ci "${TARGET_CI}")
fi

if [ "${AUTO_N_POINTS}" = "1" ]; then
  ARGS+=(--auto-n-points)
fi

if [ "${POSITIVE_CR_ONLY}" = "1" ]; then
  ARGS+=(--positive-cr-only)
fi

if [ "${FORCE_SYMMETRIC_HALF_BOX}" = "1" ]; then
  ARGS+=(--force-symmetric-half-box)
fi

srun python3 classical_solver/gep/audit_supersonic_gep_asymptotic_box.py "${ARGS[@]}"
