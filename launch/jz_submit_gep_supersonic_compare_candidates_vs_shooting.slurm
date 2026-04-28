#!/bin/bash
#SBATCH --job-name=KH_GEP_vs_shoot_sup
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
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
N_POINTS="${N_POINTS:-1001}"
TOP_K_CR="${TOP_K_CR:-6}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_gep_candidates_vs_shooting_a020_n1001_m120_130}"

echo "Comparaison directe GEP candidats vs shooting sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "N fixe: ${N_POINTS}"
echo "Top-k proches en c_r: ${TOP_K_CR}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/compare_supersonic_gep_candidates_vs_shooting.py \
  --alpha "${ALPHA_VALUE}" \
  --mach-values ${MACH_VALUES} \
  --n-points "${N_POINTS}" \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --max-abs-c 10.0 \
  --positive-ci-only \
  --top-k-cr "${TOP_K_CR}" \
  --ci-weight 2.0 \
  --shoot-cr-window 0.08 \
  --shoot-ci-window 0.03 \
  --shooting-max-iter 10 \
  --shooting-grid-size 4 \
  --output-stem "${OUTPUT_STEM}"
