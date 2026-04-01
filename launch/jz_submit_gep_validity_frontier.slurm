#!/bin/bash
#SBATCH --job-name=KH_GEP_frontier
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

echo "Job GEP validity frontier lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"

srun python3 classical_solver/gep/scan_gep_validity_frontier.py \
  --mach-min 1.10 \
  --mach-max 2.00 \
  --num-mach 10 \
  --alpha-min 0.18 \
  --alpha-max 0.24 \
  --num-alpha 5 \
  --n-values 401 481 561 641 \
  --mapping-kind pin \
  --mapping-scale 3.0 \
  --xi-max 0.99 \
  --distance-tol 0.02 \
  --previous-weight 0.3 \
  --output-stem scan_gep_validity_frontier_pin3_xi099_m110_200
