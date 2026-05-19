#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_highalpha_two_stage
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

export MACH="${MACH:-0.5}"
export ALPHA_TARGET="${ALPHA_TARGET:-0.85}"
export ALPHA_MIN="${ALPHA_MIN:-0.85}"
export ALPHA_MAX="${ALPHA_MAX:-0.85}"
export HIDDEN_DIM="${HIDDEN_DIM:-160}"

export STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-600}"
export STAGE_A_LR="${STAGE_A_LR:-1e-4}"
export STAGE_A_BALANCED_WEIGHT="${STAGE_A_BALANCED_WEIGHT:-10.0}"
export STAGE_A_BALANCED_N_XI="${STAGE_A_BALANCED_N_XI:-257}"
export STAGE_A_BALANCED_Y_MAX="${STAGE_A_BALANCED_Y_MAX:-8.0}"
export STAGE_A_BALANCED_CENTER_Y_MAX="${STAGE_A_BALANCED_CENTER_Y_MAX:-4.0}"
export STAGE_A_BALANCED_L2_WEIGHT="${STAGE_A_BALANCED_L2_WEIGHT:-1.0}"
export STAGE_A_BALANCED_OVERLAP_WEIGHT="${STAGE_A_BALANCED_OVERLAP_WEIGHT:-2.0}"
export STAGE_A_BALANCED_RHO_WEIGHT="${STAGE_A_BALANCED_RHO_WEIGHT:-1.0}"
export STAGE_A_BALANCED_U_WEIGHT="${STAGE_A_BALANCED_U_WEIGHT:-0.15}"
export STAGE_A_BALANCED_V_WEIGHT="${STAGE_A_BALANCED_V_WEIGHT:-1.0}"
export STAGE_A_BALANCED_P_WEIGHT="${STAGE_A_BALANCED_P_WEIGHT:-1.0}"

export STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-1000}"
export STAGE_B_LR="${STAGE_B_LR:-8e-5}"
export STAGE_B_MODE_SUPERVISION_WEIGHT="${STAGE_B_MODE_SUPERVISION_WEIGHT:-8.0}"
export STAGE_B_MODE_SUPERVISION_N_XI="${STAGE_B_MODE_SUPERVISION_N_XI:-257}"
export STAGE_B_MODE_SUPERVISION_Y_MAX="${STAGE_B_MODE_SUPERVISION_Y_MAX:-8.0}"
export STAGE_B_BALANCED_WEIGHT="${STAGE_B_BALANCED_WEIGHT:-5.0}"
export STAGE_B_BALANCED_N_XI="${STAGE_B_BALANCED_N_XI:-257}"
export STAGE_B_BALANCED_Y_MAX="${STAGE_B_BALANCED_Y_MAX:-8.0}"
export STAGE_B_BALANCED_CENTER_Y_MAX="${STAGE_B_BALANCED_CENTER_Y_MAX:-4.0}"
export STAGE_B_BALANCED_L2_WEIGHT="${STAGE_B_BALANCED_L2_WEIGHT:-1.0}"
export STAGE_B_BALANCED_OVERLAP_WEIGHT="${STAGE_B_BALANCED_OVERLAP_WEIGHT:-2.5}"
export STAGE_B_BALANCED_RHO_WEIGHT="${STAGE_B_BALANCED_RHO_WEIGHT:-1.0}"
export STAGE_B_BALANCED_U_WEIGHT="${STAGE_B_BALANCED_U_WEIGHT:-0.15}"
export STAGE_B_BALANCED_V_WEIGHT="${STAGE_B_BALANCED_V_WEIGHT:-1.0}"
export STAGE_B_BALANCED_P_WEIGHT="${STAGE_B_BALANCED_P_WEIGHT:-1.0}"

export WARMSTART_RUN_DIR="${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_fixed_mach_M05_highalpha_classic_balanced_full_mode_supervision}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_highalpha_classic_two_stage_repair}"

echo "Job PINN subsonique KH high-alpha classic-two-stage-repair lance sur ${SLURMD_NODENAME}"
echo "Mach=${MACH} alpha_target=${ALPHA_TARGET} bande=[${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Hypothese: stage A balanced full-mode, stage B pressure + balanced overlap avec c_i gele"
echo "Warm start=${WARMSTART_RUN_DIR}"
echo "Stage A: epochs=${STAGE_A_EPOCHS} lr=${STAGE_A_LR} balanced_w=${STAGE_A_BALANCED_WEIGHT}"
echo "Stage B: epochs=${STAGE_B_EPOCHS} lr=${STAGE_B_LR} mode_w=${STAGE_B_MODE_SUPERVISION_WEIGHT} balanced_w=${STAGE_B_BALANCED_WEIGHT}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi

srun python3 scripts/run_kh_subsonic_highalpha_classic_two_stage_repair_test.py \
  --mach "${MACH}" \
  --alpha-target "${ALPHA_TARGET}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --device cuda \
  --stage-a-epochs "${STAGE_A_EPOCHS}" \
  --stage-a-learning-rate "${STAGE_A_LR}" \
  --stage-a-balanced-weight "${STAGE_A_BALANCED_WEIGHT}" \
  --stage-a-balanced-n-xi "${STAGE_A_BALANCED_N_XI}" \
  --stage-a-balanced-y-max "${STAGE_A_BALANCED_Y_MAX}" \
  --stage-a-balanced-center-y-max "${STAGE_A_BALANCED_CENTER_Y_MAX}" \
  --stage-a-balanced-l2-weight "${STAGE_A_BALANCED_L2_WEIGHT}" \
  --stage-a-balanced-overlap-weight "${STAGE_A_BALANCED_OVERLAP_WEIGHT}" \
  --stage-a-balanced-rho-weight "${STAGE_A_BALANCED_RHO_WEIGHT}" \
  --stage-a-balanced-u-weight "${STAGE_A_BALANCED_U_WEIGHT}" \
  --stage-a-balanced-v-weight "${STAGE_A_BALANCED_V_WEIGHT}" \
  --stage-a-balanced-p-weight "${STAGE_A_BALANCED_P_WEIGHT}" \
  --stage-b-epochs "${STAGE_B_EPOCHS}" \
  --stage-b-learning-rate "${STAGE_B_LR}" \
  --stage-b-mode-supervision-weight "${STAGE_B_MODE_SUPERVISION_WEIGHT}" \
  --stage-b-mode-supervision-n-xi "${STAGE_B_MODE_SUPERVISION_N_XI}" \
  --stage-b-mode-supervision-y-max "${STAGE_B_MODE_SUPERVISION_Y_MAX}" \
  --stage-b-balanced-weight "${STAGE_B_BALANCED_WEIGHT}" \
  --stage-b-balanced-n-xi "${STAGE_B_BALANCED_N_XI}" \
  --stage-b-balanced-y-max "${STAGE_B_BALANCED_Y_MAX}" \
  --stage-b-balanced-center-y-max "${STAGE_B_BALANCED_CENTER_Y_MAX}" \
  --stage-b-balanced-l2-weight "${STAGE_B_BALANCED_L2_WEIGHT}" \
  --stage-b-balanced-overlap-weight "${STAGE_B_BALANCED_OVERLAP_WEIGHT}" \
  --stage-b-balanced-rho-weight "${STAGE_B_BALANCED_RHO_WEIGHT}" \
  --stage-b-balanced-u-weight "${STAGE_B_BALANCED_U_WEIGHT}" \
  --stage-b-balanced-v-weight "${STAGE_B_BALANCED_V_WEIGHT}" \
  --stage-b-balanced-p-weight "${STAGE_B_BALANCED_P_WEIGHT}" \
  --warmstart-run-dir "${WARMSTART_RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}"
