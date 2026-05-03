#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_highalpha_cont
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
export ALPHA_MIN="${ALPHA_MIN:-0.80}"
export ALPHA_MAX="${ALPHA_MAX:-0.86}"
export EPOCHS="${EPOCHS:-3000}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
export WARMSTART_RUN_DIR="${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_fixed_mach_M05_hybrid_alpha_sweep/alpha_0.800_high_alpha_amplitude_phase_max_wint_2}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_highalpha_continuation}"

echo "Job PINN subsonique KH high-alpha continuation lance sur ${SLURMD_NODENAME}"
echo "Mach=${MACH_VALUE} alpha_target=${ALPHA_TARGET} bande=[${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Hypothese: continuation chaude depuis alpha=0.80 sans contrainte forte de neutralite"
echo "Warm start=${WARMSTART_RUN_DIR}/model_best.pt"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun python3 scripts/run_kh_subsonic_highalpha_continuation_test.py \
  --mach "${MACH_VALUE}" \
  --alpha-target "${ALPHA_TARGET}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --device cuda \
  --warmstart-run-dir "${WARMSTART_RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}"
