#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_ci_window
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
mkdir -p model_saved

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job PINN subsonique spectral guide-window M=0.5 lance sur ${SLURMD_NODENAME}"
echo "Mach=${MACH_VALUE:-0.5} alpha dans [${ALPHA_MIN:-0.05}, ${ALPHA_MAX:-0.8}]"
echo "n_reference_alpha=${N_REFERENCE_ALPHA:-81} n_anchor_alpha=${N_ANCHOR_ALPHA:-9}"
echo "Guide epochs=${GUIDE_EPOCHS:-3000} Main epochs=${MAIN_EPOCHS:-4000} Baseline epochs=${BASELINE_EPOCHS:-4000}"
echo "Output=${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_ci_window_ablation}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun python3 scripts/run_kh_subsonic_ci_window_ablation.py \
  --mach "${MACH_VALUE:-0.5}" \
  --alpha-min "${ALPHA_MIN:-0.05}" \
  --alpha-max "${ALPHA_MAX:-0.8}" \
  --n-reference-alpha "${N_REFERENCE_ALPHA:-81}" \
  --n-anchor-alpha "${N_ANCHOR_ALPHA:-9}" \
  --guide-epochs "${GUIDE_EPOCHS:-3000}" \
  --main-epochs "${MAIN_EPOCHS:-4000}" \
  --baseline-epochs "${BASELINE_EPOCHS:-4000}" \
  --guide-hidden-dim "${GUIDE_HIDDEN_DIM:-48}" \
  --main-hidden-dim "${MAIN_HIDDEN_DIM:-64}" \
  --baseline-hidden-dim "${BASELINE_HIDDEN_DIM:-64}" \
  --guide-min-width "${GUIDE_MIN_WIDTH:-0.0025}" \
  --guide-sigma-penalty "${GUIDE_SIGMA_PENALTY:-0.03}" \
  --guide-mu-smoothness "${GUIDE_MU_SMOOTHNESS:-0.01}" \
  --guide-width-smoothness "${GUIDE_WIDTH_SMOOTHNESS:-0.01}" \
  --main-anchor-weight "${MAIN_ANCHOR_WEIGHT:-6.0}" \
  --main-smoothness-weight "${MAIN_SMOOTHNESS_WEIGHT:-0.02}" \
  --main-latent-weight "${MAIN_LATENT_WEIGHT:-0.002}" \
  --seed "${SEED:-0}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_ci_window_ablation}" \
  --device cuda
