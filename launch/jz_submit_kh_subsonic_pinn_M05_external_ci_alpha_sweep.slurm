#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_extci_asweep
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
export MACH_VALUE=0.5
export ALPHA_MIN=0.2
export ALPHA_MAX=0.8
export ALPHA_COUNT=5
export CI_MIN=0.2
export CI_MAX=0.34
export CI_COUNT=8
export EPOCHS=1500
export OUTPUT_DIR=model_saved/kh_subsonic_fixed_mach_M05_external_ci_alpha_sweep

echo "Job PINN subsonique KH sweep alpha avec recherche externe sur c_i a M=${MACH_VALUE} lance sur ${SLURMD_NODENAME}"
echo "Alpha dans [${ALPHA_MIN}, ${ALPHA_MAX}] avec ${ALPHA_COUNT} points"
echo "c_i dans [${CI_MIN}, ${CI_MAX}] avec ${CI_COUNT} points"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_alpha_sweep_external_ci.sh
