#!/bin/bash
#SBATCH --job-name=KH_shoot_visual
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
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

SUMMARY_CSV="${SUMMARY_CSV:-assets/blumen_gep/supersonic_shooting_ci_map_a020_m120_130_summary.csv}"
THRESHOLD_RATIO="${THRESHOLD_RATIO:-0.02}"
MIN_HALF_WIDTH="${MIN_HALF_WIDTH:-8.0}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_visual_validation_a020_m120_130}"

echo "Validation visuelle shooting supersonique sur ${SLURMD_NODENAME}"
echo "summary csv: ${SUMMARY_CSV}"
echo "threshold ratio: ${THRESHOLD_RATIO}"
echo "min half width: ${MIN_HALF_WIDTH}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/audit_supersonic_shooting_visual_validation.py \
  --summary-csv "${SUMMARY_CSV}" \
  --threshold-ratio "${THRESHOLD_RATIO}" \
  --min-half-width "${MIN_HALF_WIDTH}" \
  --output-stem "${OUTPUT_STEM}"
