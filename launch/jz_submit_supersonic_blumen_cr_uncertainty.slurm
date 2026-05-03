#!/bin/bash
#SBATCH --job-name=KH_blumen_cr_unc
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_cpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

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
export BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-4000}"
export REFERENCE_SUMMARY="${REFERENCE_SUMMARY:-assets/blumen_gep/supersonic_shooting_multistart_a020_m120_130_summary.csv}"
export REFERENCE_COLUMN="${REFERENCE_COLUMN:-best_shooting_cr}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_blumen_cr_uncertainty_a020_m120_130}"

echo "Audit incertitude locale Blumen c_r sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Bootstrap samples: ${BOOTSTRAP_SAMPLES}"
echo "Reference summary: ${REFERENCE_SUMMARY}"
echo "Reference column: ${REFERENCE_COLUMN}"
echo "Output stem: ${OUTPUT_STEM}"
date

srun python3 scripts/audit_supersonic_blumen_cr_uncertainty.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --reference-summary "${REFERENCE_SUMMARY}" \
  --reference-column "${REFERENCE_COLUMN}" \
  --output-stem "${OUTPUT_STEM}"
