#!/bin/bash
#SBATCH --job-name=KH_blumen_ref
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
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

ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
QUANTITY="${QUANTITY:-both}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_blumen_local_reference_a020_m120_130}"

echo "Audit local de la reference Blumen sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "quantity: ${QUANTITY}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/audit_supersonic_blumen_local_reference.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --quantity "${QUANTITY}" \
  --local-mach-half-window 0.08 \
  --local-alpha-half-window 0.08 \
  --neighbor-levels 2 \
  --output-stem "${OUTPUT_STEM}"
