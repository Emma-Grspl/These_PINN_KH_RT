#!/bin/bash
#SBATCH --job-name=KH_GEP_super_fam
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

ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
N_POINTS_VALUES="${N_POINTS_VALUES:-561 801}"
REFERENCE_MACH="${REFERENCE_MACH:-1.30}"
HIGH_CR_THRESHOLD="${HIGH_CR_THRESHOLD:-0.60}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_local_mode_families_a020_m120_130}"

echo "Diagnostic local des familles modales supersoniques sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "N values: ${N_POINTS_VALUES}"
echo "Reference Mach: ${REFERENCE_MACH}"
echo "High-cr threshold: ${HIGH_CR_THRESHOLD}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/diagnose_supersonic_local_mode_families.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --n-points-values ${N_POINTS_VALUES} \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --ci-weight 2.0 \
  --high-cr-threshold "${HIGH_CR_THRESHOLD}" \
  --reference-mach "${REFERENCE_MACH}" \
  --output-stem "${OUTPUT_STEM}"
