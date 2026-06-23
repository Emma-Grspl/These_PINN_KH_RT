#!/bin/bash
#SBATCH --job-name=KH_2D_CI0
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
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
export MACH_VALUES="${MACH_VALUES:-0.1 0.3 0.5 0.7}"
export ALPHA_MIN="${ALPHA_MIN:-0.10}"
export ALPHA_MAX="${ALPHA_MAX:-0.80}"
export ANCHOR_ALPHAS="${ANCHOR_ALPHAS:-0.10 0.30 0.55 0.80}"
export EPOCHS="${EPOCHS:-2000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export W_STAGE0_ANCHOR="${W_STAGE0_ANCHOR:-100.0}"
export W_STAGE0_MONOTONE_ALPHA="${W_STAGE0_MONOTONE_ALPHA:-1.0}"
export W_STAGE0_SMOOTH_ALPHA="${W_STAGE0_SMOOTH_ALPHA:-0.0}"
export W_STAGE0_SMOOTH_MACH="${W_STAGE0_SMOOTH_MACH:-0.0}"
export STAGE0_TARGET_MAX_ABS="${STAGE0_TARGET_MAX_ABS:-1e-3}"
export STAGE0_TARGET_MAX_REL="${STAGE0_TARGET_MAX_REL:-5e-2}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock}"

echo "Stage 0 subsonic 2D hybrid4ci on ${SLURMD_NODENAME}"
echo "Mach values=${MACH_VALUES}"
echo "Alpha range=[${ALPHA_MIN},${ALPHA_MAX}]"
echo "Anchor alphas=${ANCHOR_ALPHAS}"
echo "Output dir=${OUTPUT_DIR}"
echo "Epochs=${EPOCHS} LR=${LEARNING_RATE}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_2d_hybrid4ci_stage0_anchor_lock.sh
