#!/bin/bash
#SBATCH --job-name=KH_GEP_super_val
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
mkdir -p plot_presentation/supersonic_classique

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Validation du selecteur de branche supersonique sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"

srun python3 scripts/validate_supersonic_branch_selector.py \
  --surface-csv assets/blumen_gep/supersonic_global_m10_18_a00_03_surface.csv \
  --diagnostics-csv plot_presentation/supersonic_classique/supersonic_branch_diagnostics_top.csv \
  --output-csv plot_presentation/supersonic_classique/supersonic_branch_selector_validation.csv \
  --top-k 24 \
  --n-points 561 \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --ci-weight 2.0 \
  --branch-overlap-weight 0.35 \
  --branch-jump-cr-weight 0.45 \
  --branch-jump-ci-weight 0.20 \
  --branch-overlap-top-k 5
