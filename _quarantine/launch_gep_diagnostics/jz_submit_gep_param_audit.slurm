#!/bin/bash
#SBATCH --job-name=KH_GEP_audit
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
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

echo "Job GEP param audit lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"

COMMON_ARGS=(
  --n-values 561,641,721
  --mapping-kinds pin,cubic
  --mapping-scales 1.5,2,3,5
  --xi-max-values 0.90,0.95,0.98
  --cubic-deltas 0.1,0.2,0.4
  --top-k 30
)

echo "=== Audit 1: alpha=0.18, M=1.70 ==="
srun --exclusive python3 classical_solver/gep/audit_gep_parameter_grid.py \
  --alpha 0.18 \
  --mach 1.70 \
  "${COMMON_ARGS[@]}" \
  --output-stem audit_gep_parameter_grid_a018_m170

echo "=== Audit 2: alpha=0.18, M=1.80 ==="
srun --exclusive python3 classical_solver/gep/audit_gep_parameter_grid.py \
  --alpha 0.18 \
  --mach 1.80 \
  "${COMMON_ARGS[@]}" \
  --output-stem audit_gep_parameter_grid_a018_m180

echo "=== Audit 3: alpha=0.21, M=1.70 ==="
srun --exclusive python3 classical_solver/gep/audit_gep_parameter_grid.py \
  --alpha 0.21 \
  --mach 1.70 \
  "${COMMON_ARGS[@]}" \
  --output-stem audit_gep_parameter_grid_a021_m170
