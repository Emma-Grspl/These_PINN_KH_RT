#!/bin/bash
#SBATCH --job-name=KH_GEP_super_comp
#SBATCH --output=slurm/log/%x_%A_%a.out
#SBATCH --error=slurm/log/%x_%A_%a.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_cpu-t3
#SBATCH --array=0-3

set -euo pipefail

module purge
module load python/3.11.5

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log
mkdir -p assets/blumen_gep

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

NUM_CHUNKS=4
CHUNK_INDEX="${SLURM_ARRAY_TASK_ID}"
OUTPUT_STEM=$(printf "supersonic_global_m10_18_a00_03_composite_chunk%02d" "${CHUNK_INDEX}")

echo "Job global GEP supersonique composite chunk ${CHUNK_INDEX}/${NUM_CHUNKS} sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
echo "Output stem: ${OUTPUT_STEM}"

srun python3 classical_solver/gep/run_supersonic_mode_database_chunk.py \
  --mach-min 1.00 \
  --mach-max 1.80 \
  --num-mach 33 \
  --alpha-min 0.00 \
  --alpha-max 0.30 \
  --num-alpha 31 \
  --num-chunks "${NUM_CHUNKS}" \
  --chunk-index "${CHUNK_INDEX}" \
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
  --resume \
  --output-stem "${OUTPUT_STEM}"
