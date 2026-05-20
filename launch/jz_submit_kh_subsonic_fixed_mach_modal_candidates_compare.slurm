#!/bin/bash
#SBATCH --job-name=KH_sub1D_modal_cmp
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
module load pytorch-gpu/py3/2.1.1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/log

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "Comparaison des candidats PINN subsoniques 1D modaux sur $(hostname)"
echo "CANDIDATES=${CANDIDATES:-hybrid_8pt riccati_multibranch modefocus_lowalpha}"
echo "NUM_ALPHA_CI=${NUM_ALPHA_CI:-41}"
echo "NUM_ALPHA_MODES=${NUM_ALPHA_MODES:-21}"
echo "OVERLAY_ALPHAS=${OVERLAY_ALPHAS:-0.2 0.5 0.8}"
echo "N_Y_PINN=${N_Y_PINN:-1001}"
echo "Y_MAX=${Y_MAX:-10.0}"
echo "Y_COMMON=${Y_COMMON:-801}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT:-assets/pinn_subsonic/mach_fixed_candidates}"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

srun python3 scripts/compare_kh_subsonic_fixed_mach_modal_candidates.py \
  --output-root "${OUTPUT_ROOT:-assets/pinn_subsonic/mach_fixed_candidates}" \
  --num-alpha-ci "${NUM_ALPHA_CI:-41}" \
  --num-alpha-modes "${NUM_ALPHA_MODES:-21}" \
  --overlay-alphas ${OVERLAY_ALPHAS:-0.2 0.5 0.8} \
  --n-y-pinn "${N_Y_PINN:-1001}" \
  --y-max "${Y_MAX:-10.0}" \
  --y-common "${Y_COMMON:-801}" \
  --device cpu \
  --candidates ${CANDIDATES:-hybrid_8pt riccati_multibranch modefocus_lowalpha}
