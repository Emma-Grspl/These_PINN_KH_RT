#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_highalpha_bal_full
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@gpu
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

export MACH="${MACH:-0.5}"
export ALPHA_TARGET="${ALPHA_TARGET:-0.85}"
export ALPHA_MIN="${ALPHA_MIN:-0.85}"
export ALPHA_MAX="${ALPHA_MAX:-0.85}"
export EPOCHS="${EPOCHS:-2000}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export HIDDEN_DIM="${HIDDEN_DIM:-160}"
export BALANCED_FULL_MODE_SUPERVISION_WEIGHT="${BALANCED_FULL_MODE_SUPERVISION_WEIGHT:-10.0}"
export BALANCED_FULL_MODE_SUPERVISION_N_XI="${BALANCED_FULL_MODE_SUPERVISION_N_XI:-257}"
export BALANCED_FULL_MODE_SUPERVISION_Y_MAX="${BALANCED_FULL_MODE_SUPERVISION_Y_MAX:-8.0}"
export BALANCED_FULL_MODE_SUPERVISION_CENTER_Y_MAX="${BALANCED_FULL_MODE_SUPERVISION_CENTER_Y_MAX:-4.0}"
export BALANCED_FULL_MODE_L2_WEIGHT="${BALANCED_FULL_MODE_L2_WEIGHT:-1.0}"
export BALANCED_FULL_MODE_OVERLAP_WEIGHT="${BALANCED_FULL_MODE_OVERLAP_WEIGHT:-2.0}"
export BALANCED_FULL_MODE_RHO_WEIGHT="${BALANCED_FULL_MODE_RHO_WEIGHT:-1.0}"
export BALANCED_FULL_MODE_U_WEIGHT="${BALANCED_FULL_MODE_U_WEIGHT:-0.15}"
export BALANCED_FULL_MODE_V_WEIGHT="${BALANCED_FULL_MODE_V_WEIGHT:-1.0}"
export BALANCED_FULL_MODE_P_WEIGHT="${BALANCED_FULL_MODE_P_WEIGHT:-1.0}"
export WARMSTART_RUN_DIR="${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_fixed_mach_M05_highalpha_stepwise/stage_03_alpha_0p850}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_highalpha_classic_balanced_full_mode_supervision}"

echo "Job PINN subsonique KH high-alpha classic-balanced-full-mode supervision lance sur ${SLURMD_NODENAME}"
echo "Mach=${MACH} alpha_target=${ALPHA_TARGET} bande=[${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Hypothese: supervision classique full-mode normalisee + overlap central avec c_i gele"
echo "Warm start=${WARMSTART_RUN_DIR}"
echo "Balanced full-mode supervision: weight=${BALANCED_FULL_MODE_SUPERVISION_WEIGHT} n_xi=${BALANCED_FULL_MODE_SUPERVISION_N_XI} y_max=${BALANCED_FULL_MODE_SUPERVISION_Y_MAX} center_y_max=${BALANCED_FULL_MODE_SUPERVISION_CENTER_Y_MAX}"
echo "Field weights: rho=${BALANCED_FULL_MODE_RHO_WEIGHT} u=${BALANCED_FULL_MODE_U_WEIGHT} v=${BALANCED_FULL_MODE_V_WEIGHT} p=${BALANCED_FULL_MODE_P_WEIGHT}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi

srun python3 scripts/run_kh_subsonic_highalpha_classic_balanced_full_mode_supervision_test.py \
  --mach "${MACH}" \
  --alpha-target "${ALPHA_TARGET}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --device cuda \
  --balanced-full-mode-supervision-weight "${BALANCED_FULL_MODE_SUPERVISION_WEIGHT}" \
  --balanced-full-mode-supervision-n-xi "${BALANCED_FULL_MODE_SUPERVISION_N_XI}" \
  --balanced-full-mode-supervision-y-max "${BALANCED_FULL_MODE_SUPERVISION_Y_MAX}" \
  --balanced-full-mode-supervision-center-y-max "${BALANCED_FULL_MODE_SUPERVISION_CENTER_Y_MAX}" \
  --balanced-full-mode-l2-weight "${BALANCED_FULL_MODE_L2_WEIGHT}" \
  --balanced-full-mode-overlap-weight "${BALANCED_FULL_MODE_OVERLAP_WEIGHT}" \
  --balanced-full-mode-rho-weight "${BALANCED_FULL_MODE_RHO_WEIGHT}" \
  --balanced-full-mode-u-weight "${BALANCED_FULL_MODE_U_WEIGHT}" \
  --balanced-full-mode-v-weight "${BALANCED_FULL_MODE_V_WEIGHT}" \
  --balanced-full-mode-p-weight "${BALANCED_FULL_MODE_P_WEIGHT}" \
  --warmstart-run-dir "${WARMSTART_RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}"
