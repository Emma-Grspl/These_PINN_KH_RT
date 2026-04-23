#!/bin/bash
#SBATCH --job-name=GEP_sup_blumen_guided
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --qos=qos_cpu-t3

set -euo pipefail

module purge
module load python/3.11.5

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log
mkdir -p assets/blumen_gep

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export ALPHA_VALUES="${ALPHA_VALUES:-0.20}"
export MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
export N_POINTS="${N_POINTS:-561}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_gep_blumen_guided_a020_m120_130}"

echo "Sweep GEP supersonique guide par audit Blumen c_r/c_i sur ${SLURMD_NODENAME}"
echo "Alpha values: ${ALPHA_VALUES}"
echo "Mach values: ${MACH_VALUES}"
echo "N points: ${N_POINTS}"
echo "Output stem: ${OUTPUT_STEM}"
date

srun python3 scripts/run_supersonic_gep_blumen_guided_sweep.py \
  --alpha-values ${ALPHA_VALUES} \
  --mach-values ${MACH_VALUES} \
  --n-points "${N_POINTS}" \
  --output-stem "${OUTPUT_STEM}"
