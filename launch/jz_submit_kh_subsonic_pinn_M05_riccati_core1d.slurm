#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_riccati_core1d
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
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

echo "Job PINN subsonique KH M=0.5 Riccati core1d lance sur ${SLURMD_NODENAME}"
echo "Warm start=${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_fixed_mach_M05_riccati_lowalpha_gamma_repair_w3}"
if [ -n "${WARMSTART_CHECKPOINT:-}" ]; then
  echo "Warm checkpoint=${WARMSTART_CHECKPOINT}"
fi
echo "Epochs=${EPOCHS:-1800} LR=${LEARNING_RATE:-2e-4} CI_LR=${CI_BRANCH_LR:-5e-5} MODE_LR=${MODE_BRANCH_LR:-2e-4}"
echo "Alpha range=[${ALPHA_MIN:-0.05},${ALPHA_MAX:-0.8}] low-alpha-threshold=${LOW_ALPHA_THRESHOLD:-0.5}"
echo "Weights: ci=${CI_SUPERVISION_WEIGHT:-5.0} q=${Q_SUPERVISION_WEIGHT:-5.0} gamma=${GAMMA_SUPERVISION_WEIGHT:-3.0} anchor=${RICCATI_ANCHOR_WEIGHT:-2.0} bc_q=${BOUNDARY_Q_WEIGHT:-10.0} shoot=${SHOOTING_MATCH_WEIGHT:-12.0}"
echo "Sampling: focus_fraction=${FOCUS_FRACTION:-0.50} low_alpha_sample_fraction=${LOW_ALPHA_SAMPLE_FRACTION:-0.30}"
echo "Checkpoint every=${CHECKPOINT_EVERY:-250}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

EXTRA_ARGS=()
if [ -n "${WARMSTART_CHECKPOINT:-}" ]; then
  EXTRA_ARGS+=(--warmstart-checkpoint "${WARMSTART_CHECKPOINT}")
fi

srun python3 scripts/run_kh_subsonic_riccati_core1d.py \
  --warmstart-run-dir "${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_fixed_mach_M05_riccati_lowalpha_gamma_repair_w3}" \
  "${EXTRA_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_riccati_core1d}" \
  --epochs "${EPOCHS:-1800}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-250}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --ci-branch-lr "${CI_BRANCH_LR:-5e-5}" \
  --mode-branch-lr "${MODE_BRANCH_LR:-2e-4}" \
  --alpha-min "${ALPHA_MIN:-0.05}" \
  --alpha-max "${ALPHA_MAX:-0.8}" \
  --num-alpha-ci "${NUM_ALPHA_CI:-41}" \
  --num-alpha-modes "${NUM_ALPHA_MODES:-21}" \
  --n-y-pinn "${N_Y_PINN:-1001}" \
  --y-max "${Y_MAX:-10.0}" \
  --y-common "${Y_COMMON:-801}" \
  --low-alpha-threshold "${LOW_ALPHA_THRESHOLD:-0.5}" \
  --low-alpha-weight "${LOW_ALPHA_WEIGHT:-8.0}" \
  --low-alpha-sample-fraction "${LOW_ALPHA_SAMPLE_FRACTION:-0.30}" \
  --focus-fraction "${FOCUS_FRACTION:-0.50}" \
  --focus-half-width "${FOCUS_HALF_WIDTH:-0.04}" \
  --mode-error-threshold "${MODE_ERROR_THRESHOLD:-0.05}" \
  --ci-supervision-weight "${CI_SUPERVISION_WEIGHT:-5.0}" \
  --q-supervision-weight "${Q_SUPERVISION_WEIGHT:-5.0}" \
  --gamma-supervision-weight "${GAMMA_SUPERVISION_WEIGHT:-3.0}" \
  --riccati-anchor-weight "${RICCATI_ANCHOR_WEIGHT:-2.0}" \
  --boundary-kappa-weight "${BOUNDARY_KAPPA_WEIGHT:-5.0}" \
  --boundary-q-weight "${BOUNDARY_Q_WEIGHT:-10.0}" \
  --shooting-match-weight "${SHOOTING_MATCH_WEIGHT:-12.0}" \
  --center-kappa-weight "${CENTER_KAPPA_WEIGHT:-0.5}" \
  --center-peak-weight "${CENTER_PEAK_WEIGHT:-0.5}" \
  --overlay-alphas ${OVERLAY_ALPHAS:-0.05 0.10 0.20 0.35 0.50 0.65 0.80} \
  --anchor-alphas ${ANCHOR_ALPHAS:-0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.65 0.80} \
  --q-supervision-alphas ${Q_SUPERVISION_ALPHAS:-0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.65 0.80} \
  --gamma-supervision-alphas ${GAMMA_SUPERVISION_ALPHAS:-0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.65 0.80} \
  --device cuda
