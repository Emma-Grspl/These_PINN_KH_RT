#!/bin/bash
#SBATCH --job-name=KH_PINN_sub2d_M05_M07
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

DEFAULT_MODE_POINTS="0.10:0.50 0.25:0.50 0.65:0.50 0.10:0.55 0.25:0.55 0.65:0.55 0.10:0.60 0.25:0.60 0.65:0.60 0.10:0.65 0.25:0.65 0.65:0.65 0.10:0.70 0.25:0.70 0.65:0.70"

echo "Job PINN subsonique KH 2D bande M05-M07 lance sur ${SLURMD_NODENAME}"
echo "Warm start=${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_2d_pilot_M05_M06}"
echo "Warm checkpoint=${WARMSTART_CHECKPOINT:-auto:model_best.pt}"
echo "Warm kind=${WARMSTART_KIND:-multimach}"
echo "Alpha range=[${ALPHA_MIN:-0.05},${ALPHA_MAX:-0.75}]"
echo "Mach range=[${MACH_MIN:-0.50},${MACH_MAX:-0.70}]"
echo "Epochs=${EPOCHS:-2500} LR=${LEARNING_RATE:-5e-4} CI_LR=${CI_BRANCH_LR:-2.5e-4} MODE_LR=${MODE_BRANCH_LR:-5e-4}"
echo "Sampling: focus_fraction=${FOCUS_FRACTION:-0.55} neutral_fraction=${NEUTRAL_FRACTION:-0.20} low_alpha_fraction=${LOW_ALPHA_FRACTION:-0.15}"
echo "Mode points: ${MODE_POINTS:-${DEFAULT_MODE_POINTS}}"
echo "Skip training=${SKIP_TRAINING:-0}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
nvidia-smi || true

cmd=(
  python3 scripts/run_kh_subsonic_pinn_2d_band_M05_M07.py
  --warmstart-run-dir "${WARMSTART_RUN_DIR:-model_saved/kh_subsonic_2d_pilot_M05_M06}"
  --warmstart-kind "${WARMSTART_KIND:-multimach}"
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_2d_band_M05_M07}"
  --alpha-min "${ALPHA_MIN:-0.05}"
  --alpha-max "${ALPHA_MAX:-0.75}"
  --mach-min "${MACH_MIN:-0.50}"
  --mach-max "${MACH_MAX:-0.70}"
  --epochs "${EPOCHS:-2500}"
  --learning-rate "${LEARNING_RATE:-5e-4}"
  --ci-branch-lr "${CI_BRANCH_LR:-2.5e-4}"
  --mode-branch-lr "${MODE_BRANCH_LR:-5e-4}"
  --n-interior "${N_INTERIOR:-384}"
  --n-boundary "${N_BOUNDARY:-64}"
  --n-supervision "${N_SUPERVISION:-96}"
  --n-reference-alpha "${N_REFERENCE_ALPHA:-31}"
  --n-reference-mach "${N_REFERENCE_MACH:-9}"
  --n-audit-alpha "${N_AUDIT_ALPHA:-13}"
  --n-audit-mach "${N_AUDIT_MACH:-7}"
  --audit-every "${AUDIT_EVERY:-100}"
  --checkpoint-every "${CHECKPOINT_EVERY:-250}"
  --focus-fraction "${FOCUS_FRACTION:-0.55}"
  --neutral-fraction "${NEUTRAL_FRACTION:-0.20}"
  --low-alpha-fraction "${LOW_ALPHA_FRACTION:-0.15}"
  --upper-corner-fraction "${UPPER_CORNER_FRACTION:-0.10}"
  --lower-corner-fraction "${LOWER_CORNER_FRACTION:-0.10}"
  --focus-alpha-half-width "${FOCUS_ALPHA_HALF_WIDTH:-0.03}"
  --focus-mach-half-width "${FOCUS_MACH_HALF_WIDTH:-0.03}"
  --neutral-band-ratio "${NEUTRAL_BAND_RATIO:-0.15}"
  --low-alpha-band-width "${LOW_ALPHA_BAND_WIDTH:-0.06}"
  --upper-alpha-min "${UPPER_ALPHA_MIN:-0.65}"
  --upper-mach-min "${UPPER_MACH_MIN:-0.65}"
  --lower-alpha-max "${LOWER_ALPHA_MAX:-0.20}"
  --lower-mach-max "${LOWER_MACH_MAX:-0.53}"
  --error-threshold "${ERROR_THRESHOLD:-0.01}"
  --max-focus-points "${MAX_FOCUS_POINTS:-12}"
  --w-pde "${W_PDE:-1.0}"
  --w-bc-kappa "${W_BC_KAPPA:-10.0}"
  --w-bc-q "${W_BC_Q:-10.0}"
  --w-norm "${W_NORM:-1.0}"
  --w-phase "${W_PHASE:-1.0}"
  --w-ci-supervision "${W_CI_SUPERVISION:-5.0}"
  --ci-plot-alpha "${CI_PLOT_ALPHA:-31}"
  --ci-plot-mach "${CI_PLOT_MACH:-9}"
  --ci-n-levels "${CI_N_LEVELS:-8}"
  --mode-audit-alpha "${MODE_AUDIT_ALPHA:-5}"
  --mode-audit-mach "${MODE_AUDIT_MACH:-5}"
  --mode-n-y "${MODE_N_Y:-1001}"
  --mode-n-common "${MODE_N_COMMON:-1200}"
  --phase-threshold "${PHASE_THRESHOLD:-1e-2}"
  --device cuda
)

if [[ -n "${WARMSTART_CHECKPOINT:-}" ]]; then
  cmd+=(--warmstart-checkpoint "${WARMSTART_CHECKPOINT}")
fi

if [[ "${SKIP_TRAINING:-0}" == "1" ]]; then
  cmd+=(--skip-training)
fi

mode_points_string="${MODE_POINTS:-${DEFAULT_MODE_POINTS}}"
if [[ -n "${mode_points_string}" ]]; then
  read -r -a mode_points_array <<< "${mode_points_string}"
  cmd+=(--mode-points "${mode_points_array[@]}")
fi

srun "${cmd[@]}"
