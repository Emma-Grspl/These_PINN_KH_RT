#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_hybrid_vs_phys
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=16:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log

export MACH_VALUE="${MACH_VALUE:-0.5}"
export ALPHA_MIN="${ALPHA_MIN:-0.05}"
export ALPHA_MAX="${ALPHA_MAX:-0.85}"
export EPOCHS="${EPOCHS:-3000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_vs_physics}"

echo "Campagne comparee KH subsonique sur ${SLURMD_NODENAME}"
echo "Mach=${MACH_VALUE} alpha dans [${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Cas testes: hybride 8 points vs physique pure"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun python3 scripts/run_kh_subsonic_ci_supervision_vs_physics.py \
  --mach "${MACH_VALUE}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR}"
