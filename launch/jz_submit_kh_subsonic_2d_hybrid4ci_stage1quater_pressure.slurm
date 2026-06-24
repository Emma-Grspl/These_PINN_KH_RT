#!/bin/bash
#SBATCH --job-name=KH_2D_S1qtr
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
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_2d_hybrid4ci_stage1quater_pressure}"
export EPOCHS="${EPOCHS:-1500}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export MACH_VALUES="${MACH_VALUES:-0.1 0.3 0.5 0.7}"
export ALPHA_MIN="${ALPHA_MIN:-0.10}"
export ALPHA_MAX="${ALPHA_MAX:-0.80}"
export ANCHOR_ALPHAS="${ANCHOR_ALPHAS:-0.10 0.30 0.55 0.80}"
export N_INTERIOR="${N_INTERIOR:-256}"
export N_BOUNDARY="${N_BOUNDARY:-64}"
export N_CENTER="${N_CENTER:-192}"
export N_ALPHA_SAMPLES="${N_ALPHA_SAMPLES:-8}"
export N_MACH_SAMPLES="${N_MACH_SAMPLES:-4}"
export HIDDEN_DIM="${HIDDEN_DIM:-192}"
export DEPTH="${DEPTH:-4}"
export ACTIVATION="${ACTIVATION:-tanh}"
export FREEZE_CI="${FREEZE_CI:-1}"
export DETACH_CI_IN_MODE_BRANCH="${DETACH_CI_IN_MODE_BRANCH:-1}"
export W_PDE="${W_PDE:-1.0}"
export W_BC="${W_BC:-20.0}"
export W_GAUGE="${W_GAUGE:-100.0}"
export W_CENTER_PDE="${W_CENTER_PDE:-1.0}"
export W_CI_ANCHOR="${W_CI_ANCHOR:-1.0}"
export YMAX="${YMAX:-75.0}"
export ENVELOPE_EPS="${ENVELOPE_EPS:-1.0}"
export CENTER_WIDTH="${CENTER_WIDTH:-3.0}"
export CENTER_FRACTION="${CENTER_FRACTION:-0.6}"

echo "Stage 1quater subsonic 2D pressure-first on ${SLURMD_NODENAME}"
echo "Stage 0 checkpoint=${STAGE0_CHECKPOINT}"
echo "Output dir=${OUTPUT_DIR}"
echo "Mach values=${MACH_VALUES}"
echo "Alpha range=[${ALPHA_MIN},${ALPHA_MAX}]"
echo "Anchor alphas=${ANCHOR_ALPHAS}"
echo "Freeze ci=${FREEZE_CI} detach_ci=${DETACH_CI_IN_MODE_BRANCH}"
echo "Weights: pde=${W_PDE} bc=${W_BC} gauge=${W_GAUGE} center_pde=${W_CENTER_PDE}"
echo "ymax=${YMAX} envelope_eps=${ENVELOPE_EPS}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_2d_hybrid4ci_stage1quater_pressure.sh
