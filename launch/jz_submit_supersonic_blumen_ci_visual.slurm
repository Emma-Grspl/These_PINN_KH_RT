#!/bin/bash
#SBATCH --job-name=KH_blumen_ci_visual
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
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

export MACH_MIN="${MACH_MIN:-1.0}"
export MACH_MAX="${MACH_MAX:-2.0}"
export ALPHA_MIN="${ALPHA_MIN:-0.02}"
export ALPHA_MAX="${ALPHA_MAX:-0.50}"
export NUM_MACH="${NUM_MACH:-25}"
export NUM_ALPHA="${NUM_ALPHA:-25}"
export CI_MAX="${CI_MAX:-0.12}"
export CR_MAX="${CR_MAX:-0.35}"
export TRACKING_WEIGHT="${TRACKING_WEIGHT:-0.02}"
export OUTPUT_STEM="${OUTPUT_STEM:-supersonic_blumen_ci_visual}"

echo "Reconstruction visuelle Blumen supersonique en c_i sur ${SLURMD_NODENAME}"
echo "Mach range: ${MACH_MIN} -> ${MACH_MAX} (${NUM_MACH} points)"
echo "alpha range: ${ALPHA_MIN} -> ${ALPHA_MAX} (${NUM_ALPHA} points)"
echo "tracking_weight=${TRACKING_WEIGHT}"
echo "output stem: ${OUTPUT_STEM}"
date

srun python3 scripts/reconstruct_supersonic_blumen_ci_visual.py \
  --mach-min "${MACH_MIN}" \
  --mach-max "${MACH_MAX}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --num-mach "${NUM_MACH}" \
  --num-alpha "${NUM_ALPHA}" \
  --ci-max "${CI_MAX}" \
  --cr-max "${CR_MAX}" \
  --tracking-weight "${TRACKING_WEIGHT}" \
  --output-stem "${OUTPUT_STEM}"
