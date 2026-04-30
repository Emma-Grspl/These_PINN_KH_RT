#!/bin/bash
#SBATCH --job-name=KH_blumen_audit
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
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_blumen_eigencondition_audit_a020_m120_130}"
REFERENCE_SUMMARY="${REFERENCE_SUMMARY:-assets/blumen_gep/supersonic_shooting_multistart_a020_m120_130_summary.csv}"

echo "Audit eigenconditions / asymptotiques supersoniques sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Reference summary: ${REFERENCE_SUMMARY}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/audit_supersonic_blumen_eigenconditions.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --match-y 1.0 \
  --use-mapping \
  --mapping-scale 5.0 \
  --reference-summary "${REFERENCE_SUMMARY}" \
  --map-cr-half-window 0.05 \
  --map-ci-half-window 0.02 \
  --map-n-cr 25 \
  --map-n-ci 21 \
  --tail-fraction 0.80 \
  --match-band-width 0.20 \
  --output-stem "${OUTPUT_STEM}"
