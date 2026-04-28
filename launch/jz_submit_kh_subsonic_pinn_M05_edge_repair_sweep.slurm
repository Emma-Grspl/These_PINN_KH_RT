#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_EDGE
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
export ALPHA_VALUES="${ALPHA_VALUES:-0.05 0.85}"
export EPOCHS="${EPOCHS:-3000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export HIDDEN_DIM="${HIDDEN_DIM:-160}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_edge_repair_sweep}"

echo "Job PINN subsonique KH edge-repair lance sur ${SLURMD_NODENAME}"
echo "Mach=${MACH_VALUE} alphas=${ALPHA_VALUES}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun python3 scripts/run_kh_subsonic_edge_repair_sweep.py \
  --mach "${MACH_VALUE}" \
  --alpha-values ${ALPHA_VALUES} \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR}"
