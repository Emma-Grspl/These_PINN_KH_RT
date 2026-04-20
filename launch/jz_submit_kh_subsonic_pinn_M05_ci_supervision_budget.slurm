#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_ci_budget
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log
mkdir -p model_saved

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export DEVICE=cuda
export MACH_VALUE=0.5
export ALPHA_MIN=0.05
export ALPHA_MAX=0.85
export EPOCHS=3000
export COUNTS=4,8,16,32,64,128
export CI_MAE_TARGET=0.02
export OUTPUT_DIR=model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_budget

echo "Job PINN subsonique KH budget minimal de points classiques pour c_i lance sur ${SLURMD_NODENAME}"
echo "Mach=${MACH_VALUE} alpha dans [${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Budgets testes: ${COUNTS}"
echo "Seuil cible sur ci_mae: ${CI_MAE_TARGET}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_ci_supervision_budget.sh
