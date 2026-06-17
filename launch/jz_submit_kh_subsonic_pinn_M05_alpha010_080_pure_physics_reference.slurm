#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_pure_a010_080
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

export DEVICE=cuda
export MACH_VALUE="${MACH_VALUE:-0.5}"
export ALPHA_MIN="${ALPHA_MIN:-0.10}"
export ALPHA_MAX="${ALPHA_MAX:-0.80}"
export EPOCHS="${EPOCHS:-5000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_riccati_pure_physics_reference}"

echo "Job PINN subsonique KH M=${MACH_VALUE} Riccati pure physics reference alpha=[${ALPHA_MIN},${ALPHA_MAX}] lance sur ${SLURMD_NODENAME}"
echo "Output dir=${OUTPUT_DIR}"
echo "Weights: bc_k=${W_BC_KAPPA:-10.0} bc_q=${W_BC_Q:-20.0} center_k=${W_RICCATI_CENTER_KAPPA:-5.0} center_peak=${W_RICCATI_CENTER_PEAK:-2.0} band_k=${W_RICCATI_BOUNDARY_BAND_KAPPA:-2.0} band_q=${W_RICCATI_BOUNDARY_BAND_Q:-8.0} ci_low=${W_CI_LOW_ALPHA_ZERO:-10.0} ci_smooth=${W_CI_SMOOTHNESS:-0.5}"
echo "Epochs=${EPOCHS} LR=${LEARNING_RATE}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_M05_alpha010_080_pure_physics_reference.sh
