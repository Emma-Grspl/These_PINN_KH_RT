#!/bin/bash
#SBATCH --job-name=KH_shoot_Mach_cont
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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

ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
ANCHOR_MACH="${ANCHOR_MACH:-1.20}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_mach_continuation_a020_m120_130}"

echo "Continuation shooting supersonique en Mach sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Mach ancre: ${ANCHOR_MACH}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/track_supersonic_shooting_mach_continuation.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --anchor-mach "${ANCHOR_MACH}" \
  --match-y 1.0 \
  --use-mapping \
  --mapping-scale 5.0 \
  --anchor-cr-half-window 0.08 \
  --anchor-ci-half-window 0.03 \
  --cont-cr-half-window 0.03 \
  --cont-ci-half-window 0.015 \
  --retry-growth 1.75 \
  --max-retries 4 \
  --max-iter 10 \
  --grid-size 4 \
  --output-stem "${OUTPUT_STEM}"
