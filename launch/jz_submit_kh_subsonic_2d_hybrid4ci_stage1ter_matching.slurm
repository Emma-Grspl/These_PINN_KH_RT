#!/bin/bash
#SBATCH --job-name=KH_2D_S1ter
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
export N_INTERIOR="${N_INTERIOR:-512}"
export N_BOUNDARY="${N_BOUNDARY:-96}"
export N_CENTER="${N_CENTER:-256}"
export CENTER_WIDTH="${CENTER_WIDTH:-2.0}"
export CENTER_FRACTION="${CENTER_FRACTION:-0.5}"
export N_ALPHA_SAMPLES="${N_ALPHA_SAMPLES:-12}"
export N_MACH_SAMPLES="${N_MACH_SAMPLES:-4}"
export EPOCHS="${EPOCHS:-8000}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_2d_hybrid4ci_stage1ter_riccati_matching}"
export FREEZE_CI="${FREEZE_CI:-1}"
export DETACH_CI_IN_MODE_BRANCH="${DETACH_CI_IN_MODE_BRANCH:-1}"
export W_PDE="${W_PDE:-1.0}"
export W_BC_KAPPA="${W_BC_KAPPA:-20.0}"
export W_BC_Q="${W_BC_Q:-60.0}"
export W_MATCH="${W_MATCH:-1.0}"
export W_CENTER_PDE="${W_CENTER_PDE:-1.0}"
export W_NORM="${W_NORM:-0.0}"
export W_PHASE="${W_PHASE:-0.0}"
export W_CI_ANCHOR="${W_CI_ANCHOR:-1.0}"
export MATCH_WARMUP_EPOCHS="${MATCH_WARMUP_EPOCHS:-1000}"
export SHOOT_YMAX="${SHOOT_YMAX:-40.0}"
export SHOOT_STEPS="${SHOOT_STEPS:-512}"
export MATCH_Y_VALUES="${MATCH_Y_VALUES:--1.0 -0.5 0.0 0.5 1.0}"

echo "Stage 1ter subsonic 2D Riccati matching on ${SLURMD_NODENAME}"
echo "Stage 0 checkpoint=${STAGE0_CHECKPOINT}"
echo "Mach values=${MACH_VALUES}"
echo "Alpha range=[${ALPHA_MIN},${ALPHA_MAX}]"
echo "Anchor alphas=${ANCHOR_ALPHAS}"
echo "Match y values=${MATCH_Y_VALUES}"
echo "Output dir=${OUTPUT_DIR}"
echo "Freeze ci=${FREEZE_CI} detach_ci=${DETACH_CI_IN_MODE_BRANCH}"
echo "Weights: pde=${W_PDE} bc_kappa=${W_BC_KAPPA} bc_q=${W_BC_Q} match=${W_MATCH} center=${W_CENTER_PDE}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_2d_hybrid4ci_stage1ter_matching.sh
