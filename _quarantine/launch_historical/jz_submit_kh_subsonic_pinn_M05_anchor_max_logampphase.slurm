#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_MX_LOGAMP
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

echo "Job PINN subsonique KH M=0.5 sweep alpha anchor=max log-amplitude/phase lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
nvidia-smi || true

srun python3 scripts/train_kh_subsonic_pinn.py \
  --mach 0.5 \
  --alpha-min 0.05 \
  --alpha-max 0.85 \
  --epochs 5000 \
  --audit-every 100 \
  --checkpoint-every 500 \
  --mapping-scale 3.0 \
  --n-reference-alpha 81 \
  --n-audit-alpha 41 \
  --n-mode-audit-alpha 9 \
  --n-mode-audit-y 801 \
  --n-anchor-alpha 48 \
  --n-norm-interior 320 \
  --focus-fraction 0.8 \
  --focus-half-width 0.02 \
  --error-threshold 0.01 \
  --mode-error-threshold 0.45 \
  --max-focus-points 6 \
  --anchor-strategy max \
  --anchor-half-width 0.10 \
  --anchor-max-candidates 257 \
  --mode-center-fraction 0.8 \
  --mode-center-half-width 0.20 \
  --mode-representation log_amplitude_phase \
  --hidden-dim 160 \
  --w-pde 1.0 \
  --w-bc 10.0 \
  --w-norm 1.5 \
  --w-integral-norm 2.0 \
  --w-phase 4.0 \
  --w-ci-supervision 2.0 \
  --audit-ci-weight 5.0 \
  --audit-mode-weight 1.0 \
  --classic-n-points 561 \
  --classic-mapping-scale 3.0 \
  --classic-xi-max 0.99 \
  --device cuda \
  --output-dir model_saved/kh_subsonic_fixed_mach_M05_anchor_max_logampphase
