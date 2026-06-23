#!/bin/bash
#SBATCH --job-name=KH_stage1bis_assets
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
module load pytorch-gpu/py3/2.1.1

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log
mkdir -p assets/pinn_subsonic/stage1bis_article

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MACH_VALUE="${MACH_VALUE:-0.5}"
export ALPHA_MIN="${ALPHA_MIN:-0.10}"
export ALPHA_MAX="${ALPHA_MAX:-0.80}"
export NUM_ALPHA="${NUM_ALPHA:-71}"
export MODAL_ALPHAS="${MODAL_ALPHAS:-0.30 0.50 0.70}"
export OUTPUT_DIR="${OUTPUT_DIR:-assets/pinn_subsonic/stage1bis_article}"
export SAVE_PDF="${SAVE_PDF:-1}"

echo "Render Stage1bis article assets on ${SLURMD_NODENAME}"
echo "MACH=${MACH_VALUE}"
echo "ALPHA=[${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "NUM_ALPHA=${NUM_ALPHA}"
echo "MODAL_ALPHAS=${MODAL_ALPHAS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
date

ARGS=(
  --mach "${MACH_VALUE}"
  --alpha-min "${ALPHA_MIN}"
  --alpha-max "${ALPHA_MAX}"
  --num-alpha "${NUM_ALPHA}"
  --modal-alphas ${MODAL_ALPHAS}
  --output-dir "${OUTPUT_DIR}"
  --device cpu
)

if [ "${SAVE_PDF}" = "1" ]; then
  ARGS+=(--save-pdf)
fi

srun python scripts/render_kh_subsonic_stage1bis_article_assets.py "${ARGS[@]}"

date
