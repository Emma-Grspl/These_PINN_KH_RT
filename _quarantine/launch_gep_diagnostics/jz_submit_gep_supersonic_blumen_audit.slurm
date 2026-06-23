#!/bin/bash
#SBATCH --job-name=KH_GEP_super_audit
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
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

OUTPUT_STEM="audit_supersonic_blumen_points"

echo "Job audit GEP supersonique Blumen sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 classical_solver/gep/audit_supersonic_blumen_points.py \
  --families ci_level,ci_special,cr_special \
  --n-values 561 \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --distance-tol 0.02 \
  --ci-weight 2.0 \
  --previous-weight 0.6 \
  --cr-min 0.03 \
  --cr-max-scale 0.5 \
  --cr-max-floor 0.35 \
  --cr-max-cap 0.7 \
  --ci-min 0.001 \
  --ci-max 0.12 \
  --shooting-max-iter 10 \
  --output-stem "${OUTPUT_STEM}"
