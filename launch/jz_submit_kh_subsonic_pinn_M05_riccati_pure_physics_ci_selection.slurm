#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_riccati_phys_ci
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job PINN subsonique KH M=${MACH_VALUE:-0.5} Riccati pure physics + ci selection lance sur ${SLURMD_NODENAME}"
echo "Alpha range=[${ALPHA_MIN:-0.05},${ALPHA_MAX:-0.85}]"
echo "Epochs=${EPOCHS:-5000} LR=${LEARNING_RATE:-1e-3}"
echo "Weights: bc_k=${W_BC_KAPPA:-10.0} bc_q=${W_BC_Q:-20.0} center_k=${W_RICCATI_CENTER_KAPPA:-5.0} center_peak=${W_RICCATI_CENTER_PEAK:-2.0} band_k=${W_RICCATI_BOUNDARY_BAND_KAPPA:-2.0} band_q=${W_RICCATI_BOUNDARY_BAND_Q:-8.0} shoot=${W_RICCATI_SHOOTING_MATCH:-20.0} ci_opt=${W_RICCATI_CI_LOCAL_MIN:-10.0}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

export DEVICE=cuda
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_riccati_pure_physics_ci_selection}"

srun bash scripts/run_kh_subsonic_riccati_pure_physics_ci_selection.sh
