#!/bin/bash
#SBATCH --job-name=KH_S1_M05_ci8
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

mkdir -p slurm/log model_saved

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export DEVICE="${DEVICE:-cuda}"
export MACH_VALUE="${MACH_VALUE:-0.5}"
export ALPHA_MIN="${ALPHA_MIN:-0.10}"
export ALPHA_MAX="${ALPHA_MAX:-0.80}"
export EPOCHS="${EPOCHS:-5000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export MODE_BRANCH_LR="${MODE_BRANCH_LR:-1e-3}"
export CI_SUP_COUNT=8
export CI_SUP_FIXED_ALPHAS="${CI_SUP_FIXED_ALPHAS:-0.100 0.180 0.260 0.340 0.450 0.560 0.680 0.800}"
export STAGE0_MODEL_PATH="${STAGE0_MODEL_PATH:-model_saved/kh_subsonic_M05_alpha010_080_ci_stage0_anchor_lock/hybrid_ci8/model_best.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_M05_alpha010_080_hybrid_stage1_mode_from_ci_stage0/hybrid_ci8}"

echo "Stage 1 mode-only subsonique M=${MACH_VALUE} ci=${CI_SUP_COUNT} alpha=[${ALPHA_MIN},${ALPHA_MAX}] sur ${SLURMD_NODENAME}"
echo "Stage 0 checkpoint=${STAGE0_MODEL_PATH}"
echo "Anchors=${CI_SUP_FIXED_ALPHAS}"
echo "Output dir=${OUTPUT_DIR}"
echo "Epochs=${EPOCHS} LR=${LEARNING_RATE} mode_lr=${MODE_BRANCH_LR}"
echo "CI is frozen; classic ci supervision is disabled; classic references are audit-only."
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_M05_alpha010_080_hybrid_stage1_mode_from_ci_stage0.sh
