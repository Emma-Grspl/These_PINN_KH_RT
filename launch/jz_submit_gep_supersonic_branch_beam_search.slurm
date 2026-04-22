#!/bin/bash
#SBATCH --job-name=GEP_sup_branch_beam
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
REFERENCE_MACH="${REFERENCE_MACH:-1.30}"
N_POINTS="${N_POINTS:-561}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_branch_beam_search_a020_m120_130}"

echo "Tracking multi-branches supersonique par beam search sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Reference Mach: ${REFERENCE_MACH}"
echo "N points: ${N_POINTS}"
echo "Output stem: ${OUTPUT_STEM}"
date

srun python scripts/track_supersonic_branch_beam_search.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --reference-mach "${REFERENCE_MACH}" \
  --n-points "${N_POINTS}" \
  --high-cr-threshold 0.60 \
  --beam-width 6 \
  --branch-top-k 4 \
  --candidate-top-k 40 \
  --w-prev-overlap 1.30 \
  --w-ref-overlap 0.90 \
  --w-prev-cr 0.70 \
  --w-prev-ci 0.40 \
  --w-centroid 0.40 \
  --w-spread 0.25 \
  --w-phase-span 0.25 \
  --w-shooting 0.20 \
  --w-blumen-ci 0.20 \
  --w-cr-floor 0.20 \
  --soft-cr-floor 0.55 \
  --w-low-ci 1.20 \
  --neutral-ci-threshold 0.0075 \
  --output-stem "${OUTPUT_STEM}"
