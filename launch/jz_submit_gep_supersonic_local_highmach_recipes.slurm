#!/bin/bash
#SBATCH --job-name=KH_GEP_highmach_local
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

echo "Job GEP local haute-Mach lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"

for MACH in 1.6 1.7 1.8; do
  echo
  echo "=== Sweep local recette PIN pour M=${MACH} ==="
  srun --exclusive --ntasks=1 python3 classical_solver/gep/adaptive_continuation_sweep_gep.py \
    --mach "${MACH}" \
    --alpha-min 0.18 \
    --alpha-max 0.21 \
    --num-points 4 \
    --n-values 561 641 721 \
    --mapping-kind pin \
    --mapping-scale 1.5 \
    --cubic-delta 0.2 \
    --xi-max 0.90 \
    --distance-tol 0.02 \
    --ci-weight 2.0 \
    --previous-weight 0.6 \
    --output-stem highmach_local_pin15_xi090

  echo
  echo "=== Sweep local recette CUBIC pour M=${MACH} ==="
  srun --exclusive --ntasks=1 python3 classical_solver/gep/adaptive_continuation_sweep_gep.py \
    --mach "${MACH}" \
    --alpha-min 0.18 \
    --alpha-max 0.21 \
    --num-points 4 \
    --n-values 561 641 721 \
    --mapping-kind cubic \
    --mapping-scale 5.0 \
    --cubic-delta 0.4 \
    --xi-max 0.98 \
    --distance-tol 0.02 \
    --ci-weight 2.0 \
    --previous-weight 0.6 \
    --output-stem highmach_local_cubic5_d04_xi098
done
