#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_mode_err_alpha
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
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

export BASE_DIR="${BASE_DIR:-model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_vs_physics}"
export OUTPUT_DIR="${OUTPUT_DIR:-plot_presentation/subsonic_pinn/ci_supervision_vs_physics}"
export NUM_ALPHA="${NUM_ALPHA:-17}"
export N_Y_PINN="${N_Y_PINN:-1001}"
export DEVICE="${DEVICE:-cuda}"

echo "Diagnostic erreur de mode vs alpha sur ${SLURMD_NODENAME}"
echo "Base dir: ${BASE_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Nombre de points alpha: ${NUM_ALPHA}"
echo "Device: ${DEVICE}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun python3 scripts/plot_kh_subsonic_mode_error_vs_alpha.py \
  --base-dir "${BASE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-alpha "${NUM_ALPHA}" \
  --n-y-pinn "${N_Y_PINN}" \
  --device "${DEVICE}"
