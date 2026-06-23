#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_modecmp
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export RUN_A_DIR="${RUN_A_DIR:-model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_vs_physics/hybrid_8pt}"
export RUN_B_DIR="${RUN_B_DIR:-model_saved/kh_subsonic_fixed_mach_M05_modefocus_lowalpha}"
export RUN_A_LABEL="${RUN_A_LABEL:-PINN hybride, 8 points}"
export RUN_B_LABEL="${RUN_B_LABEL:-PINN mode-focus low-alpha}"
export OUTPUT_DIR="${OUTPUT_DIR:-plot_presentation/subsonic_pinn/modefocus_comparison}"
export OUTPUT_STEM="${OUTPUT_STEM:-mode_error_vs_alpha_8pt_vs_modefocus}"
export NUM_ALPHA="${NUM_ALPHA:-17}"
export N_Y_PINN="${N_Y_PINN:-1001}"
export DEVICE="${DEVICE:-cuda}"

echo "Comparaison erreur de mode vs alpha sur ${SLURMD_NODENAME}"
echo "Run A: ${RUN_A_DIR}"
echo "Run B: ${RUN_B_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Output stem: ${OUTPUT_STEM}"
echo "Nombre de points alpha: ${NUM_ALPHA}"
date
nvidia-smi || true

srun python scripts/plot_kh_subsonic_mode_error_vs_alpha.py \
  --run-a-dir "${RUN_A_DIR}" \
  --run-b-dir "${RUN_B_DIR}" \
  --run-a-label "${RUN_A_LABEL}" \
  --run-b-label "${RUN_B_LABEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-stem "${OUTPUT_STEM}" \
  --num-alpha "${NUM_ALPHA}" \
  --n-y-pinn "${N_Y_PINN}" \
  --device "${DEVICE}"
