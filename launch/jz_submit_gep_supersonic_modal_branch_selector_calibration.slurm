#!/bin/bash
#SBATCH --job-name=KH_GEP_modal_calib
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

export ALPHA_VALUE="${ALPHA_VALUE:-0.20}"
export MACH_VALUES="${MACH_VALUES:-1.20 1.25 1.275 1.30}"
export REFERENCE_MACH="${REFERENCE_MACH:-1.30}"
export N_POINTS="${N_POINTS:-561}"
export MAPPING_KIND="${MAPPING_KIND:-pin}"
export MAPPING_SCALE="${MAPPING_SCALE:-1.5}"
export CUBIC_DELTA="${CUBIC_DELTA:-0.2}"
export XI_MAX="${XI_MAX:-0.90}"
export CI_WEIGHT="${CI_WEIGHT:-2.0}"
export HIGH_CR_THRESHOLD="${HIGH_CR_THRESHOLD:-0.60}"
export CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-80}"
export DISTANCE_TOL="${DISTANCE_TOL:-0.02}"
export OUTPUT_STEM_PREFIX="${OUTPUT_STEM_PREFIX:-supersonic_modal_branch_selector_calib_a020_m120_130}"

echo "Calibration du selecteur modal de branche supersonique sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Mach values: ${MACH_VALUES}"
echo "Reference Mach: ${REFERENCE_MACH}"
echo "Output stem prefix: ${OUTPUT_STEM_PREFIX}"

srun bash scripts/run_supersonic_modal_branch_selector_calibration.sh
