#!/bin/bash
#SBATCH --job-name=KH_GEP_modal_branch
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
REFERENCE_MACH="${REFERENCE_MACH:-1.30}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_modal_branch_selector_a020_m120_130}"

echo "Diagnostic selecteur modal de branche supersonique sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Reference Mach: ${REFERENCE_MACH}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/diagnose_supersonic_modal_branch_selector.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --reference-mach "${REFERENCE_MACH}" \
  --n-points 561 \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --ci-weight 2.0 \
  --high-cr-threshold 0.60 \
  --candidate-top-k 80 \
  --distance-tol 0.02 \
  --w-shooting 1.0 \
  --w-reference-overlap 1.0 \
  --w-previous-overlap 0.6 \
  --w-centroid 0.5 \
  --w-spread 0.3 \
  --w-phase-span 0.3 \
  --w-cr-jump 0.2 \
  --w-ci-jump 0.1 \
  --output-stem "${OUTPUT_STEM}"
