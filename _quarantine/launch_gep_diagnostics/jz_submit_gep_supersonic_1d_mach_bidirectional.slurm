#!/bin/bash
#SBATCH --job-name=KH_GEP_super_1d_m_bi
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
MACH_ANCHOR="${MACH_ANCHOR:-1.40}"
MACH_MIN="${MACH_MIN:-1.00}"
MACH_MAX="${MACH_MAX:-1.80}"
MACH_STEP="${MACH_STEP:-0.025}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_1d_mach_bidirectional_a020}"

DOWN_STEM="${OUTPUT_STEM}_down"
UP_STEM="${OUTPUT_STEM}_up"

echo "Sweep 1D supersonique bidirectionnel en Mach sur ${SLURMD_NODENAME}"
echo "alpha fixe: ${ALPHA_VALUE}"
echo "Anchor Mach: ${MACH_ANCHOR}"
echo "Mach in [${MACH_MIN}, ${MACH_MAX}] pas ${MACH_STEP}"
echo "Output stem: ${OUTPUT_STEM}"

MACH_VALUES_DOWN="$(python3 - <<PY
import numpy as np
anchor=float(${MACH_ANCHOR})
vmin=float(${MACH_MIN})
step=float(${MACH_STEP})
vals=np.arange(anchor, vmin-step*0.5, -step)
print(' '.join(f'{v:.10f}' for v in vals))
PY
)"

MACH_VALUES_UP="$(python3 - <<PY
import numpy as np
anchor=float(${MACH_ANCHOR})
vmax=float(${MACH_MAX})
step=float(${MACH_STEP})
vals=np.arange(anchor, vmax+step*0.5, step)
print(' '.join(f'{v:.10f}' for v in vals))
PY
)"

srun python3 classical_solver/gep/build_supersonic_mode_database.py \
  --mach-min "${MACH_MIN}" \
  --mach-max "${MACH_MAX}" \
  --num-mach 1 \
  --mach-values ${MACH_VALUES_DOWN} \
  --alpha-min "${ALPHA_VALUE}" \
  --alpha-max "${ALPHA_VALUE}" \
  --num-alpha 1 \
  --n-values 561 \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --distance-tol 0.02 \
  --ci-weight 2.0 \
  --previous-weight 0.6 \
  --branch-overlap-weight 0.35 \
  --branch-jump-cr-weight 0.45 \
  --branch-jump-ci-weight 0.20 \
  --branch-overlap-top-k 5 \
  --y-points 561 \
  --output-stem "${DOWN_STEM}"

srun python3 classical_solver/gep/build_supersonic_mode_database.py \
  --mach-min "${MACH_MIN}" \
  --mach-max "${MACH_MAX}" \
  --num-mach 1 \
  --mach-values ${MACH_VALUES_UP} \
  --alpha-min "${ALPHA_VALUE}" \
  --alpha-max "${ALPHA_VALUE}" \
  --num-alpha 1 \
  --n-values 561 \
  --mapping-kind pin \
  --mapping-scale 1.5 \
  --cubic-delta 0.2 \
  --xi-max 0.90 \
  --distance-tol 0.02 \
  --ci-weight 2.0 \
  --previous-weight 0.6 \
  --branch-overlap-weight 0.35 \
  --branch-jump-cr-weight 0.45 \
  --branch-jump-ci-weight 0.20 \
  --branch-overlap-top-k 5 \
  --y-points 561 \
  --output-stem "${UP_STEM}"

python3 classical_solver/gep/merge_supersonic_mode_database_chunks.py \
  --chunk-stems "${DOWN_STEM}" "${UP_STEM}" \
  --output-stem "${OUTPUT_STEM}"
