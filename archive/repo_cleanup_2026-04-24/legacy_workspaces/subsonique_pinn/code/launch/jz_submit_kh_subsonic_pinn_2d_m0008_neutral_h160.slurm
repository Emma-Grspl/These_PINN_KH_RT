#!/bin/bash
#SBATCH --job-name=KH_PINN_sub2d_M08_h160
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

echo "Job PINN subsonique KH 2D (M in [0,0.8], h=160) lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
nvidia-smi || true

srun python3 scripts/train_kh_subsonic_pinn_2d.py \
  --alpha-min 0.05 \
  --alpha-max 0.60 \
  --mach-min 0.0 \
  --mach-max 0.8 \
  --epochs 5000 \
  --learning-rate 1e-3 \
  --hidden-dim 160 \
  --mode-depth 4 \
  --ci-depth 2 \
  --activation tanh \
  --audit-every 100 \
  --checkpoint-every 500 \
  --mapping-scale 3.0 \
  --n-reference-alpha 41 \
  --n-reference-mach 17 \
  --n-audit-alpha 17 \
  --n-audit-mach 9 \
  --focus-fraction 0.6 \
  --neutral-fraction 0.2 \
  --focus-alpha-half-width 0.03 \
  --focus-mach-half-width 0.05 \
  --neutral-band-ratio 0.15 \
  --error-threshold 0.02 \
  --max-focus-points 12 \
  --device cuda \
  --output-dir model_saved/kh_subsonic_2d_local_m0008_jz_neutral_h160
