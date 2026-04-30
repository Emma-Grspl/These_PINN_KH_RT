#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_lowalpha_target
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
export ALPHA_TARGET="${ALPHA_TARGET:-0.05}"
export ALPHA_MIN="${ALPHA_MIN:-0.05}"
export ALPHA_MAX="${ALPHA_MAX:-0.20}"
export EPOCHS="${EPOCHS:-4000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_lowalpha_targeted}"

echo "Job PINN subsonique KH low-alpha cible lance sur ${SLURMD_NODENAME}"
echo "Mach=${MACH_VALUE} alpha_target=${ALPHA_TARGET} bande=[${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Hypothese: branche riccati/q-supervision dediee au bas-alpha"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun python3 scripts/run_kh_subsonic_targeted_regime_test.py \
  --regime low_alpha \
  --mach "${MACH_VALUE}" \
  --alpha-target "${ALPHA_TARGET}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR}"
