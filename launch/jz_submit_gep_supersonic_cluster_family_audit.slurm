#!/bin/bash
#SBATCH --job-name=GEP_sup_cluster_audit
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
export MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
export REFERENCE_MACH="${REFERENCE_MACH:-1.30}"
export N_POINTS="${N_POINTS:-561}"
export CLUSTER_STEM="${CLUSTER_STEM:-supersonic_family_clustering_a020_m120_130}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_cluster_family_audit_a020_m120_130}"

echo "Audit des familles clusterisees supersoniques sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Reference Mach: ${REFERENCE_MACH}"
echo "N points: ${N_POINTS}"
echo "Cluster stem: ${CLUSTER_STEM}"
echo "Output stem: ${OUTPUT_STEM}"
date

srun python3 scripts/audit_supersonic_cluster_families.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --reference-mach "${REFERENCE_MACH}" \
  --n-points "${N_POINTS}" \
  --cluster-stem "${CLUSTER_STEM}" \
  --output-stem "${OUTPUT_STEM}"
