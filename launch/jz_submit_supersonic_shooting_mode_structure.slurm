#!/bin/bash
#SBATCH --job-name=KH_shoot_mode
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
CENTER_WINDOW="${CENTER_WINDOW:-8.0}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_shooting_mode_structure_a020_m120_130}"

echo "Audit structure modale shooting supersonique sur ${SLURMD_NODENAME}"
echo "summary csv: ${SUMMARY_CSV}"
echo "center window: ${CENTER_WINDOW}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 scripts/audit_supersonic_shooting_mode_structure.py \
  --summary-csv "${SUMMARY_CSV}" \
  --center-window "${CENTER_WINDOW}" \
  --common-grid-size 801 \
  --output-stem "${OUTPUT_STEM}"
