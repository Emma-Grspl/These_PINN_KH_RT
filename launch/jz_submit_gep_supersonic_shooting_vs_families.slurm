#!/bin/bash
#SBATCH --job-name=KH_GEP_shoot_audit
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
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

export ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
export MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
export REFERENCE_MACH="${REFERENCE_MACH:-1.30}"
export N_POINTS="${N_POINTS:-561}"
export MAPPING_KIND="${MAPPING_KIND:-pin}"
export MAPPING_SCALE="${MAPPING_SCALE:-1.5}"
export CUBIC_DELTA="${CUBIC_DELTA:-0.2}"
export XI_MAX="${XI_MAX:-0.90}"
export CI_WEIGHT="${CI_WEIGHT:-2.0}"
export HIGH_CR_THRESHOLD="${HIGH_CR_THRESHOLD:-0.60}"
export SHOOTING_MAX_ITER="${SHOOTING_MAX_ITER:-10}"
export SHOOTING_GRID_SIZE="${SHOOTING_GRID_SIZE:-4}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_vs_gep_families_a020_m120_130}"

echo "Audit shooting-vs-familles GEP supersonique sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Reference Mach: ${REFERENCE_MACH}"
echo "N points: ${N_POINTS}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/audit_supersonic_shooting_vs_gep_families.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --reference-mach "${REFERENCE_MACH}" \
  --n-points "${N_POINTS}" \
  --mapping-kind "${MAPPING_KIND}" \
  --mapping-scale "${MAPPING_SCALE}" \
  --cubic-delta "${CUBIC_DELTA}" \
  --xi-max "${XI_MAX}" \
  --ci-weight "${CI_WEIGHT}" \
  --high-cr-threshold "${HIGH_CR_THRESHOLD}" \
  --shooting-max-iter "${SHOOTING_MAX_ITER}" \
  --shooting-grid-size "${SHOOTING_GRID_SIZE}" \
  --output-stem "${OUTPUT_STEM}"
