#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_riccati_exp1
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

echo "Job PINN subsonique KH M=0.5 Riccati exp1 q-loss + low-alpha lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi

export DEVICE=cuda
export OUTPUT_DIR=model_saved/kh_subsonic_fixed_mach_M05_riccati_exp1_q_lowalpha

srun bash scripts/run_kh_subsonic_exp1_q_lowalpha.sh
