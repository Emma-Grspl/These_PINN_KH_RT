#!/bin/bash
#SBATCH --job-name=KH_Blumensup
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

mkdir -p slurm/log
mkdir -p assets/blumen_shooting_supersonic

# Matplotlib et fontconfig doivent écrire dans un espace temporaire sur Jean Zay.
export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

# Le solveur actuel est principalement CPU-bound ; on borne les bibliothèques BLAS.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job lance sur ${SLURMD_NODENAME}"
nvidia-smi || true

srun python3 classical_solver/supersonic/reconstruct_blumen_supersonic_shooting.py \
  --mach-min 1.0 \
  --mach-max 1.4 \
  --alpha-min 0.14 \
  --alpha-max 0.34 \
  --num-mach 5 \
  --num-alpha 5 \
  --tracking-weight 0.02 \
  --output-stem supersonic_local_jz
