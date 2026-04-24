#!/bin/bash
#SBATCH --job-name=KH_GEP_super_raw_blumen
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
N_POINTS_VALUES="${N_POINTS_VALUES:-561 801 1001}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_raw_spectrum_vs_blumen_a020_m120_130}"
CR_BOX_TOL="${CR_BOX_TOL:-0.03}"
CI_BOX_TOL="${CI_BOX_TOL:-0.015}"
TOP_K_NEAREST="${TOP_K_NEAREST:-12}"

echo "Diagnostic du spectre GEP brut supersonique contre Blumen sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "N values: ${N_POINTS_VALUES}"
echo "Boite Blumen: dcr<=${CR_BOX_TOL}, dci<=${CI_BOX_TOL}"
echo "Top nearest exportes: ${TOP_K_NEAREST}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/diagnose_supersonic_raw_spectrum_vs_blumen.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --n-points-values ${N_POINTS_VALUES} \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --max-abs-c 10.0 \
  --positive-ci-only \
  --cr-box-tol "${CR_BOX_TOL}" \
  --ci-box-tol "${CI_BOX_TOL}" \
  --top-k-nearest "${TOP_K_NEAREST}" \
  --output-stem "${OUTPUT_STEM}"
