#!/bin/bash
#SBATCH --job-name=KH_PINN_sub2d_ci_mode
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
mkdir -p model_saved

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job PINN subsonique KH 2D ci+mode lance sur ${SLURMD_NODENAME}"
echo "alpha: ${ALPHA_MIN:-0.05} -> ${ALPHA_MAX:-0.85}"
echo "mach: ${MACH_MIN:-0.0} -> ${MACH_MAX:-0.5}"
echo "mode points: ${MODE_POINTS:-default}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
nvidia-smi || true

srun python3 scripts/run_kh_subsonic_pinn_2d_ci_mode_campaign.py \
  --alpha-min "${ALPHA_MIN:-0.05}" \
  --alpha-max "${ALPHA_MAX:-0.85}" \
  --mach-min "${MACH_MIN:-0.0}" \
  --mach-max "${MACH_MAX:-0.5}" \
  --epochs "${EPOCHS:-5000}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --hidden-dim "${HIDDEN_DIM:-160}" \
  --mode-hidden-dim "${MODE_HIDDEN_DIM:-160}" \
  --ci-hidden-dim "${CI_HIDDEN_DIM:-80}" \
  --mode-depth "${MODE_DEPTH:-4}" \
  --ci-depth "${CI_DEPTH:-2}" \
  --activation "${ACTIVATION:-tanh}" \
  --mode-representation "${MODE_REPRESENTATION:-riccati}" \
  --audit-every "${AUDIT_EVERY:-100}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-500}" \
  --mapping-scale "${MAPPING_SCALE:-3.0}" \
  --n-reference-alpha "${N_REFERENCE_ALPHA:-41}" \
  --n-reference-mach "${N_REFERENCE_MACH:-11}" \
  --n-audit-alpha "${N_AUDIT_ALPHA:-17}" \
  --n-audit-mach "${N_AUDIT_MACH:-6}" \
  --focus-fraction "${FOCUS_FRACTION:-0.6}" \
  --neutral-fraction "${NEUTRAL_FRACTION:-0.2}" \
  --low-alpha-fraction "${LOW_ALPHA_FRACTION:-0.10}" \
  --upper-corner-fraction "${UPPER_CORNER_FRACTION:-0.15}" \
  --lower-corner-fraction "${LOWER_CORNER_FRACTION:-0.10}" \
  --focus-alpha-half-width "${FOCUS_ALPHA_HALF_WIDTH:-0.03}" \
  --focus-mach-half-width "${FOCUS_MACH_HALF_WIDTH:-0.05}" \
  --neutral-band-ratio "${NEUTRAL_BAND_RATIO:-0.15}" \
  --low-alpha-band-width "${LOW_ALPHA_BAND_WIDTH:-0.06}" \
  --upper-alpha-min "${UPPER_ALPHA_MIN:-0.75}" \
  --upper-mach-min "${UPPER_MACH_MIN:-0.40}" \
  --lower-alpha-max "${LOWER_ALPHA_MAX:-0.25}" \
  --lower-mach-max "${LOWER_MACH_MAX:-0.05}" \
  --error-threshold "${ERROR_THRESHOLD:-0.02}" \
  --max-focus-points "${MAX_FOCUS_POINTS:-12}" \
  --w-bc-kappa "${W_BC_KAPPA:-10.0}" \
  --w-bc-q "${W_BC_Q:-10.0}" \
  --w-ci-supervision "${W_CI_SUPERVISION:-5.0}" \
  --separate-branch-optimizers \
  --detach-ci-in-mode-branch \
  --ci-branch-lr "${CI_BRANCH_LR:-1e-3}" \
  --mode-branch-lr "${MODE_BRANCH_LR:-1e-3}" \
  --ci-plot-alpha "${CI_PLOT_ALPHA:-41}" \
  --ci-plot-mach "${CI_PLOT_MACH:-11}" \
  --ci-n-levels "${CI_N_LEVELS:-8}" \
  --mode-audit-alpha "${MODE_AUDIT_ALPHA:-7}" \
  --mode-audit-mach "${MODE_AUDIT_MACH:-5}" \
  ${MODE_POINTS:+--mode-points ${MODE_POINTS}} \
  --mode-n-y "${MODE_N_Y:-1001}" \
  --mode-n-common "${MODE_N_COMMON:-1200}" \
  --phase-threshold "${PHASE_THRESHOLD:-1e-2}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_alphamach_ci_mode_campaign}"
