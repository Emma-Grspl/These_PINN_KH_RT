#!/bin/bash
#SBATCH --job-name=GEP_sup_branch_cont
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
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

ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
REFERENCE_MACH="${REFERENCE_MACH:-1.30}"
N_POINTS="${N_POINTS:-561}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_branch_continuation_blumen_a020_m120_130}"

echo "Tracking de branche supersonique par continuation sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Reference Mach: ${REFERENCE_MACH}"
echo "N points: ${N_POINTS}"
echo "Output stem: ${OUTPUT_STEM}"
date

srun python scripts/track_supersonic_branch_continuation.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --reference-mach "${REFERENCE_MACH}" \
  --n-points "${N_POINTS}" \
  --high-cr-threshold 0.60 \
  --candidate-top-k 40 \
  --w-prev-overlap 1.80 \
  --w-ref-overlap 1.20 \
  --w-prev-cr 0.80 \
  --w-prev-ci 0.45 \
  --w-centroid 0.55 \
  --w-spread 0.35 \
  --w-phase-span 0.35 \
  --w-shooting 0.10 \
  --w-blumen-ci 0.15 \
  --w-cr-floor 0.25 \
  --soft-cr-floor 0.55 \
  --output-stem "${OUTPUT_STEM}"
