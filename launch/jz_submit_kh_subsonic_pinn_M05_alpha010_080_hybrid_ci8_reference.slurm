#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_hyb_ci8
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
export MACH_VALUE="${MACH_VALUE:-0.5}"
export ALPHA_MIN="${ALPHA_MIN:-0.10}"
export ALPHA_MAX="${ALPHA_MAX:-0.80}"
export EPOCHS="${EPOCHS:-5000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export CI_SUP_COUNT=8
export CI_SUP_FIXED_ALPHAS="0.100 0.193 0.287 0.380 0.520 0.613 0.707 0.800"
export W_CI_SUPERVISION="${W_CI_SUPERVISION:-5.0}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/hybrid_ci8_fixed}"

echo "Job PINN subsonique KH M=${MACH_VALUE} hybride sparse ci=8 alpha=[${ALPHA_MIN},${ALPHA_MAX}] lance sur ${SLURMD_NODENAME}"
echo "Fixed ci anchors=${CI_SUP_FIXED_ALPHAS}"
echo "Output dir=${OUTPUT_DIR}"
echo "Epochs=${EPOCHS} LR=${LEARNING_RATE} w_ci=${W_CI_SUPERVISION}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_M05_alpha010_080_ci_sparse_reference.sh
