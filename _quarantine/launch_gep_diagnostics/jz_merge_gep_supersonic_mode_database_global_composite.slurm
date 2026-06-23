#!/bin/bash
#SBATCH --job-name=KH_GEP_super_cmp_merge
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
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

echo "Merge base GEP supersonique globale composite sur ${SLURMD_NODENAME}"

srun python3 classical_solver/gep/merge_supersonic_mode_database_chunks.py \
  --chunk-stems \
    supersonic_global_m10_18_a00_03_composite_chunk00 \
    supersonic_global_m10_18_a00_03_composite_chunk01 \
    supersonic_global_m10_18_a00_03_composite_chunk02 \
    supersonic_global_m10_18_a00_03_composite_chunk03 \
  --output-stem supersonic_global_m10_18_a00_03_composite
