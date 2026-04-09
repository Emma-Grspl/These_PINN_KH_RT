#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_riccati_2stage
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
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

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "Job PINN subsonique KH M=0.5 Riccati two-stage lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi

srun python3 scripts/train_kh_subsonic_pinn.py \
  --mach 0.5 \
  --alpha-min 0.05 \
  --alpha-max 0.85 \
  --epochs 5000 \
  --learning-rate 1e-3 \
  --hidden-dim 160 \
  --mode-depth 4 \
  --ci-depth 2 \
  --activation tanh \
  --mapping-scale 3.0 \
  --n-interior 512 \
  --n-boundary 64 \
  --n-alpha-supervision 128 \
  --n-anchor-alpha 32 \
  --n-norm-interior 256 \
  --n-reference-alpha 81 \
  --n-audit-alpha 21 \
  --n-mode-audit-alpha 7 \
  --n-mode-audit-y 801 \
  --audit-every 100 \
  --checkpoint-every 500 \
  --stage-split-epoch 2000 \
  --stage2-freeze-ci \
  --stage1-w-ci-supervision 5.0 \
  --stage2-w-ci-supervision 0.0 \
  --stage1-neutral-fraction 0.2 \
  --stage2-neutral-fraction 0.0 \
  --focus-fraction 0.6 \
  --focus-half-width 0.03 \
  --neutral-half-width 0.04 \
  --error-threshold 0.01 \
  --mode-error-threshold 0.12 \
  --max-focus-points 8 \
  --anchor-strategy point \
  --w-pde 1.0 \
  --w-bc-kappa 10.0 \
  --w-bc-q 20.0 \
  --w-ci-supervision 5.0 \
  --audit-ci-weight 10.0 \
  --audit-env-weight 1.0 \
  --audit-phase-weight 0.5 \
  --audit-peak-weight 0.25 \
  --phase-mask-fraction 0.15 \
  --classic-n-points 561 \
  --classic-mapping-scale 3.0 \
  --classic-xi-max 0.99 \
  --mode-representation riccati \
  --device cuda \
  --output-dir model_saved/kh_subsonic_fixed_mach_M05_riccati_twostage
