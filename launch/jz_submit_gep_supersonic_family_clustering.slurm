#!/bin/bash
#SBATCH --job-name=GEP_sup_family_cluster
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
export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
REFERENCE_MACH="${REFERENCE_MACH:-1.30}"
N_POINTS="${N_POINTS:-561}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_family_clustering_a020_m120_130}"

echo "Clustering des familles supersoniques sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Reference Mach: ${REFERENCE_MACH}"
echo "N points: ${N_POINTS}"
echo "Output stem: ${OUTPUT_STEM}"
date

srun python scripts/cluster_supersonic_mode_families.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --reference-mach "${REFERENCE_MACH}" \
  --n-points "${N_POINTS}" \
  --high-cr-threshold 0.60 \
  --valid-cr-max 1.20 \
  --cluster-cr-window 0.08 \
  --cluster-ci-window 0.01 \
  --cluster-overlap-threshold 0.92 \
  --cluster-centroid-window 0.40 \
  --cluster-phase-window 0.35 \
  --w-match-overlap 1.40 \
  --w-match-cr 0.90 \
  --w-match-ci 0.45 \
  --w-match-centroid 0.35 \
  --w-match-spread 0.20 \
  --w-match-phase 0.20 \
  --output-stem "${OUTPUT_STEM}"
