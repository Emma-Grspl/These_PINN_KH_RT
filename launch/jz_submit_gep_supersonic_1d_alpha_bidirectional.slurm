#!/bin/bash
#SBATCH --job-name=KH_GEP_super_1d_a_bi
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

MACH_VALUE="${MACH_VALUE:-1.40}"
ALPHA_ANCHOR="${ALPHA_ANCHOR:-0.20}"
ALPHA_MIN="${ALPHA_MIN:-0.00}"
ALPHA_MAX="${ALPHA_MAX:-0.30}"
ALPHA_STEP="${ALPHA_STEP:-0.01}"
OUTPUT_STEM="${OUTPUT_STEM:-supersonic_1d_alpha_bidirectional_m140}"

DOWN_STEM="${OUTPUT_STEM}_down"
UP_STEM="${OUTPUT_STEM}_up"

echo "Sweep 1D supersonique bidirectionnel en alpha sur ${SLURMD_NODENAME}"
echo "Mach fixe: ${MACH_VALUE}"
echo "Anchor alpha: ${ALPHA_ANCHOR}"
echo "alpha in [${ALPHA_MIN}, ${ALPHA_MAX}] pas ${ALPHA_STEP}"
echo "Output stem: ${OUTPUT_STEM}"

ALPHA_VALUES_DOWN="$(python3 - <<PY
import numpy as np
anchor=float(${ALPHA_ANCHOR})
amin=float(${ALPHA_MIN})
step=float(${ALPHA_STEP})
vals=np.arange(anchor, amin-step*0.5, -step)
print(' '.join(f'{v:.10f}' for v in vals))
PY
)"

ALPHA_VALUES_UP="$(python3 - <<PY
import numpy as np
anchor=float(${ALPHA_ANCHOR})
amax=float(${ALPHA_MAX})
step=float(${ALPHA_STEP})
vals=np.arange(anchor, amax+step*0.5, step)
print(' '.join(f'{v:.10f}' for v in vals))
PY
)"

srun python3 classical_solver/gep/build_supersonic_mode_database.py \
  --mach-min "${MACH_VALUE}" \
  --mach-max "${MACH_VALUE}" \
  --num-mach 1 \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --num-alpha 1 \
  --alpha-values ${ALPHA_VALUES_DOWN} \
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
  --mach-min "${MACH_VALUE}" \
  --mach-max "${MACH_VALUE}" \
  --num-mach 1 \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --num-alpha 1 \
  --alpha-values ${ALPHA_VALUES_UP} \
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

