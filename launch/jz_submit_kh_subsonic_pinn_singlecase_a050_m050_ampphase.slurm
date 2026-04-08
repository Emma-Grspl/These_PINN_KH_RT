#!/bin/bash
#SBATCH --job-name=KH_PINN_A050_M050_AP
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

echo "Job PINN subsonique KH single-case alpha=0.5 M=0.5 amplitude-phase lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
nvidia-smi || true

srun python3 scripts/train_kh_subsonic_pinn.py \
  --mach 0.5 \
  --alpha-min 0.5 \
  --alpha-max 0.5 \
  --epochs 5000 \
  --audit-every 100 \
  --checkpoint-every 500 \
  --mapping-scale 3.0 \
  --n-reference-alpha 1 \
  --n-audit-alpha 1 \
  --n-mode-audit-alpha 1 \
  --n-mode-audit-y 1201 \
  --n-anchor-alpha 64 \
  --n-norm-interior 384 \
  --focus-fraction 0.0 \
  --focus-half-width 0.0 \
  --error-threshold 0.0 \
  --mode-error-threshold 0.0 \
  --max-focus-points 0 \
  --anchor-half-width 0.10 \
  --mode-center-fraction 0.9 \
  --mode-center-half-width 0.18 \
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
  --mode-representation amplitude_phase \
  --device cuda \
  --output-dir model_saved/kh_subsonic_singlecase_a050_m050_ampphase
