#!/bin/bash
#SBATCH --job-name=KH_shoot_ci_map
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
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

ALPHA_VALUES="${ALPHA_VALUES:-0.20}"
MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_ci_map_a020_m120_130}"

echo "Audit shooting supersonique c_i sur ${SLURMD_NODENAME}"
echo "alpha values: ${ALPHA_VALUES}"
echo "Mach values: ${MACH_VALUES}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/audit_supersonic_shooting_ci_map.py \
  --alpha-values ${ALPHA_VALUES} \
  --mach-values ${MACH_VALUES} \
  --match-y 1.0 \
  --use-mapping \
  --mapping-scale 5.0 \
  --cr-half-windows 0.015 0.03 0.06 0.10 \
  --ci-half-windows 0.008 0.015 0.03 \
  --retry-growth 1.75 \
  --max-retries 3 \
  --max-iter 10 \
  --grid-size 4 \
  --ci-weight 4.0 \
  --cr-weight 0.35 \
  --continuity-weight 0.20 \
  --output-stem "${OUTPUT_STEM}"
