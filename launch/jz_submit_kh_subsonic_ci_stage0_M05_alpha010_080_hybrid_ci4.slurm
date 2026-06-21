#!/bin/bash
#SBATCH --job-name=KH_CI0_M05_ci4
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
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
export EPOCHS="${EPOCHS:-1000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-2}"
export CI_SUP_COUNT=4
export CI_SUP_FIXED_ALPHAS="${CI_SUP_FIXED_ALPHAS:-0.100 0.300 0.550 0.800}"
export W_STAGE0_ANCHOR="${W_STAGE0_ANCHOR:-100.0}"
export W_STAGE0_MONOTONE="${W_STAGE0_MONOTONE:-1.0}"
export W_STAGE0_SMOOTH="${W_STAGE0_SMOOTH:-0.0}"
export STAGE0_LBFGS_STEPS="${STAGE0_LBFGS_STEPS:-500}"
export STAGE0_LBFGS_LR="${STAGE0_LBFGS_LR:-0.5}"
export STAGE0_TARGET_MAX_ABS="${STAGE0_TARGET_MAX_ABS:-1e-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_M05_alpha010_080_ci_stage0_anchor_lock/hybrid_ci4}"

echo "Stage 0 CI-only subsonique M=${MACH_VALUE} hybride ci=${CI_SUP_COUNT} alpha=[${ALPHA_MIN},${ALPHA_MAX}] sur ${SLURMD_NODENAME}"
echo "Anchors=${CI_SUP_FIXED_ALPHAS}"
echo "Output dir=${OUTPUT_DIR}"
echo "Epochs=${EPOCHS} LR=${LEARNING_RATE}"
echo "Weights: anchor=${W_STAGE0_ANCHOR} monotone=${W_STAGE0_MONOTONE} smooth=${W_STAGE0_SMOOTH}"
echo "LBFGS: steps=${STAGE0_LBFGS_STEPS} lr=${STAGE0_LBFGS_LR}"
echo "Target max abs=${STAGE0_TARGET_MAX_ABS}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_M05_alpha010_080_ci_stage0_anchor_lock.sh
