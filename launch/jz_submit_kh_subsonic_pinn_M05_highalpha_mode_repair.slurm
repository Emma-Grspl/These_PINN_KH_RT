#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_highalpha_fix
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

export MACH_VALUE="${MACH_VALUE:-0.5}"
export ALPHA_TARGET="${ALPHA_TARGET:-0.85}"
export ALPHA_MIN="${ALPHA_MIN:-0.845}"
export ALPHA_MAX="${ALPHA_MAX:-0.85}"
export EPOCHS="${EPOCHS:-2000}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
export WARMSTART_RUN_DIR="${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_fixed_mach_M05_highalpha_stepwise/stage_02_alpha_0p840}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_highalpha_mode_repair}"

echo "Job PINN subsonique KH high-alpha mode-repair lance sur ${SLURMD_NODENAME}"
echo "Mach=${MACH_VALUE} alpha_target=${ALPHA_TARGET} bande=[${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Hypothese: c_i gele, reparation de mode depuis le checkpoint 0.84"
echo "Warm start=${WARMSTART_RUN_DIR}/model_best.pt"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun python3 scripts/run_kh_subsonic_highalpha_mode_repair_test.py \
  --mach "${MACH_VALUE}" \
  --alpha-target "${ALPHA_TARGET}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --device cuda \
  --warmstart-run-dir "${WARMSTART_RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}"
