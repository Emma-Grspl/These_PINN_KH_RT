#!/bin/bash
#SBATCH --job-name=KH_2D_S1
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
export STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt}"
export MACH_VALUES="${MACH_VALUES:-0.1 0.3 0.5 0.7}"
export ALPHA_MIN="${ALPHA_MIN:-0.10}"
export ALPHA_MAX="${ALPHA_MAX:-0.80}"
export ANCHOR_ALPHAS="${ANCHOR_ALPHAS:-0.10 0.30 0.55 0.80}"
export N_INTERIOR="${N_INTERIOR:-128}"
export N_BOUNDARY="${N_BOUNDARY:-32}"
export N_ALPHA_SAMPLES="${N_ALPHA_SAMPLES:-8}"
export N_MACH_SAMPLES="${N_MACH_SAMPLES:-4}"
export EPOCHS="${EPOCHS:-5000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0}"
export FREEZE_CI="${FREEZE_CI:-1}"

echo "Stage 1 subsonic 2D hybrid4ci on ${SLURMD_NODENAME}"
echo "Stage 0 checkpoint=${STAGE0_CHECKPOINT}"
echo "Mach values=${MACH_VALUES}"
echo "Alpha range=[${ALPHA_MIN},${ALPHA_MAX}]"
echo "Anchor alphas=${ANCHOR_ALPHAS}"
echo "Output dir=${OUTPUT_DIR}"
echo "Freeze ci=${FREEZE_CI}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.sh
