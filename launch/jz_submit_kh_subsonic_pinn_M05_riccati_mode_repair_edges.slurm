#!/bin/bash
#SBATCH --job-name=KH_PINN_M05_riccati_mode_edges
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

echo "Job PINN subsonique KH M=0.5 Riccati mode-repair edges lance sur ${SLURMD_NODENAME}"
echo "Warm start=${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_fixed_mach_M05_riccati_core1d_resume_from750_finalize}"
if [ -n "${WARMSTART_CHECKPOINT:-}" ]; then
  echo "Warm checkpoint=${WARMSTART_CHECKPOINT}"
fi
echo "Epochs=${EPOCHS:-600} LR=${LEARNING_RATE:-1e-4} MODE_LR=${MODE_BRANCH_LR:-1e-4}"
echo "Alpha range=[${ALPHA_MIN:-0.2},${ALPHA_MAX:-0.8}]"
echo "Edges: low<=${EDGE_LOW_THRESHOLD:-0.30} high>=${EDGE_HIGH_THRESHOLD:-0.70}"
echo "Sampling: focus_fraction=${FOCUS_FRACTION:-0.40} low_edge_fraction=${LOW_ALPHA_SAMPLE_FRACTION:-0.30} high_edge_fraction=${HIGH_ALPHA_SAMPLE_FRACTION:-0.30}"
echo "Weights: q=${Q_SUPERVISION_WEIGHT:-5.0} gamma=${GAMMA_SUPERVISION_WEIGHT:-3.0} anchor=${RICCATI_ANCHOR_WEIGHT:-2.0} full_mode=${BALANCED_FULL_MODE_WEIGHT:-1.0}"
echo "Checkpoint every=${CHECKPOINT_EVERY:-100} audit every=${AUDIT_EVERY:-100}"
echo "Skip training=${SKIP_TRAINING:-0}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

EXTRA_ARGS=()
if [ -n "${WARMSTART_CHECKPOINT:-}" ]; then
  EXTRA_ARGS+=(--warmstart-checkpoint "${WARMSTART_CHECKPOINT}")
fi
if [ "${SKIP_TRAINING:-0}" = "1" ]; then
  EXTRA_ARGS+=(--skip-training)
fi

srun python3 scripts/run_kh_subsonic_riccati_mode_repair_edges.py \
  --warmstart-run-dir "${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_fixed_mach_M05_riccati_core1d_resume_from750_finalize}" \
  "${EXTRA_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_riccati_mode_repair_edges}" \
  --epochs "${EPOCHS:-600}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-100}" \
  --audit-every "${AUDIT_EVERY:-100}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --mode-branch-lr "${MODE_BRANCH_LR:-1e-4}" \
  --alpha-min "${ALPHA_MIN:-0.2}" \
  --alpha-max "${ALPHA_MAX:-0.8}" \
  --num-alpha-ci "${NUM_ALPHA_CI:-41}" \
  --num-alpha-modes "${NUM_ALPHA_MODES:-25}" \
  --n-y-pinn "${N_Y_PINN:-1001}" \
  --y-max "${Y_MAX:-10.0}" \
  --y-common "${Y_COMMON:-801}" \
  --edge-low-threshold "${EDGE_LOW_THRESHOLD:-0.30}" \
  --edge-high-threshold "${EDGE_HIGH_THRESHOLD:-0.70}" \
  --edge-low-weight "${EDGE_LOW_WEIGHT:-3.0}" \
  --edge-high-weight "${EDGE_HIGH_WEIGHT:-3.0}" \
  --edge-low-audit-fraction "${EDGE_LOW_AUDIT_FRACTION:-0.35}" \
  --edge-high-audit-fraction "${EDGE_HIGH_AUDIT_FRACTION:-0.35}" \
  --low-alpha-sample-fraction "${LOW_ALPHA_SAMPLE_FRACTION:-0.30}" \
  --high-alpha-sample-fraction "${HIGH_ALPHA_SAMPLE_FRACTION:-0.30}" \
  --focus-fraction "${FOCUS_FRACTION:-0.40}" \
  --focus-half-width "${FOCUS_HALF_WIDTH:-0.03}" \
  --mode-error-threshold "${MODE_ERROR_THRESHOLD:-0.05}" \
  --audit-ci-weight "${AUDIT_CI_WEIGHT:-0.25}" \
  --audit-p-weight "${AUDIT_P_WEIGHT:-4.0}" \
  --audit-env-weight "${AUDIT_ENV_WEIGHT:-2.0}" \
  --audit-phase-weight "${AUDIT_PHASE_WEIGHT:-2.0}" \
  --audit-peak-weight "${AUDIT_PEAK_WEIGHT:-1.0}" \
  --q-supervision-weight "${Q_SUPERVISION_WEIGHT:-5.0}" \
  --gamma-supervision-weight "${GAMMA_SUPERVISION_WEIGHT:-3.0}" \
  --riccati-anchor-weight "${RICCATI_ANCHOR_WEIGHT:-2.0}" \
  --boundary-kappa-weight "${BOUNDARY_KAPPA_WEIGHT:-5.0}" \
  --boundary-q-weight "${BOUNDARY_Q_WEIGHT:-10.0}" \
  --shooting-match-weight "${SHOOTING_MATCH_WEIGHT:-12.0}" \
  --center-kappa-weight "${CENTER_KAPPA_WEIGHT:-0.5}" \
  --center-peak-weight "${CENTER_PEAK_WEIGHT:-0.5}" \
  --balanced-full-mode-weight "${BALANCED_FULL_MODE_WEIGHT:-1.0}" \
  --balanced-full-mode-u-weight "${BALANCED_FULL_MODE_U_WEIGHT:-0.25}" \
  --balanced-full-mode-v-weight "${BALANCED_FULL_MODE_V_WEIGHT:-1.0}" \
  --balanced-full-mode-p-weight "${BALANCED_FULL_MODE_P_WEIGHT:-1.0}" \
  --balanced-full-mode-rho-weight "${BALANCED_FULL_MODE_RHO_WEIGHT:-1.0}" \
  --overlay-alphas ${OVERLAY_ALPHAS:-0.20 0.25 0.30 0.50 0.70 0.75 0.80} \
  --anchor-alphas ${ANCHOR_ALPHAS:-0.20 0.25 0.30 0.50 0.70 0.75 0.80} \
  --q-supervision-alphas ${Q_SUPERVISION_ALPHAS:-0.20 0.25 0.30 0.50 0.70 0.75 0.80} \
  --gamma-supervision-alphas ${GAMMA_SUPERVISION_ALPHAS:-0.20 0.25 0.30 0.50 0.70 0.75 0.80} \
  --balanced-full-mode-alphas ${BALANCED_FULL_MODE_ALPHAS:-0.20 0.25 0.30 0.50 0.70 0.75 0.80} \
  --device cuda
