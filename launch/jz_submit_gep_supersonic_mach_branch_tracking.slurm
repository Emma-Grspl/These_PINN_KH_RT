#!/bin/bash
#SBATCH --job-name=KH_GEP_super_track_m
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
MACH_ANCHOR="${MACH_ANCHOR:-1.40}"
MACH_MIN="${MACH_MIN:-1.00}"
MACH_MAX="${MACH_MAX:-1.80}"
MACH_STEP="${MACH_STEP:-0.025}"
CRITICAL_MACH_MIN="${CRITICAL_MACH_MIN:-1.20}"
CRITICAL_MACH_MAX="${CRITICAL_MACH_MAX:-1.35}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_mach_branch_tracking_a020_anchor140}"

echo "Diagnostic tracking de branche Mach supersonique sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach anchor: ${MACH_ANCHOR}"
echo "Mach in [${MACH_MIN}, ${MACH_MAX}] pas ${MACH_STEP}"
echo "Critical Mach window: [${CRITICAL_MACH_MIN}, ${CRITICAL_MACH_MAX}]"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/diagnose_supersonic_mach_branch_tracking.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-anchor "${MACH_ANCHOR}" \
  --mach-min "${MACH_MIN}" \
  --mach-max "${MACH_MAX}" \
  --mach-step "${MACH_STEP}" \
  --n-points 561 \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --distance-tol 0.02 \
  --ci-weight 2.0 \
  --previous-weight 0.6 \
  --branch-overlap-weight 0.35 \
  --branch-anchor-weight 0.60 \
  --branch-jump-cr-weight 0.45 \
  --branch-jump-ci-weight 0.20 \
  --branch-overlap-top-k 5 \
  --critical-mach-min "${CRITICAL_MACH_MIN}" \
  --critical-mach-max "${CRITICAL_MACH_MAX}" \
  --critical-top-k 6 \
  --output-stem "${OUTPUT_STEM}"
