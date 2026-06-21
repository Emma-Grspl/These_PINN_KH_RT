#!/bin/bash
#SBATCH --job-name=KH_PINN_single_match_ter
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

export DEVICE=cuda
export ALPHA_VALUE="${ALPHA_VALUE:-0.65}"
export MACH_VALUE="${MACH_VALUE:-0.60}"
export W_BC_KAPPA="${W_BC_KAPPA:-10.0}"
export W_BC_Q="${W_BC_Q:-25.0}"
export W_RICCATI_CENTER_KAPPA="${W_RICCATI_CENTER_KAPPA:-1.0}"
export W_RICCATI_CENTER_PEAK="${W_RICCATI_CENTER_PEAK:-0.0}"
export W_RICCATI_BOUNDARY_BAND_KAPPA="${W_RICCATI_BOUNDARY_BAND_KAPPA:-0.5}"
export W_RICCATI_BOUNDARY_BAND_Q="${W_RICCATI_BOUNDARY_BAND_Q:-2.0}"
export W_RICCATI_SHOOTING_MATCH="${W_RICCATI_SHOOTING_MATCH:-20.0}"
export W_RICCATI_SHOOTING_PATH="${W_RICCATI_SHOOTING_PATH:-5.0}"
export RICCATI_SHOOTING_PATH_POINTS="${RICCATI_SHOOTING_PATH_POINTS:-17}"
export RICCATI_SHOOTING_XI_BOUNDARY="${RICCATI_SHOOTING_XI_BOUNDARY:-0.985}"
export RICCATI_SHOOTING_MATCH_START_EPOCH="${RICCATI_SHOOTING_MATCH_START_EPOCH:-200}"
export RICCATI_SHOOTING_PATH_XI_BOUNDARY="${RICCATI_SHOOTING_PATH_XI_BOUNDARY:-0.94}"
export RICCATI_SHOOTING_PATH_START_EPOCH="${RICCATI_SHOOTING_PATH_START_EPOCH:-600}"
export RICCATI_SHOOTING_PATH_EVERY="${RICCATI_SHOOTING_PATH_EVERY:-10}"
export W_RICCATI_CI_LOCAL_MIN="${W_RICCATI_CI_LOCAL_MIN:-5.0}"
export RICCATI_CI_LOCAL_MIN_START_EPOCH="${RICCATI_CI_LOCAL_MIN_START_EPOCH:-900}"
export RICCATI_CI_LOCAL_MIN_DELTA_ABS="${RICCATI_CI_LOCAL_MIN_DELTA_ABS:-0.002}"
export RICCATI_CI_LOCAL_MIN_DELTA_REL="${RICCATI_CI_LOCAL_MIN_DELTA_REL:-0.02}"
export RICCATI_CI_LOCAL_MIN_MARGIN="${RICCATI_CI_LOCAL_MIN_MARGIN:-1e-4}"
export INITIAL_CI="${INITIAL_CI:-0.2}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export SEPARATE_BRANCH_OPTIMIZERS="${SEPARATE_BRANCH_OPTIMIZERS:-1}"
export DETACH_CI_IN_MODE_BRANCH="${DETACH_CI_IN_MODE_BRANCH:-1}"
export CI_BRANCH_LR="${CI_BRANCH_LR:-5e-5}"
export MODE_BRANCH_LR="${MODE_BRANCH_LR:-2e-4}"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  alpha_tag="$(printf '%0.3f' "${ALPHA_VALUE}" | tr -d '.')"
  mach_tag="$(printf '%0.3f' "${MACH_VALUE}" | tr -d '.')"
  export OUTPUT_DIR="model_saved/kh_subsonic_singlecase_phaseA_ter_a${alpha_tag}_m${mach_tag}_scalar_ci_matching"
fi

echo "Job PINN subsonique KH single-case Riccati pure physics Phase A-ter alpha=${ALPHA_VALUE} M=${MACH_VALUE} lance sur ${SLURMD_NODENAME}"
echo "Output dir=${OUTPUT_DIR}"
echo "Weights: bc_k=${W_BC_KAPPA} bc_q=${W_BC_Q} center_k=${W_RICCATI_CENTER_KAPPA} center_peak=${W_RICCATI_CENTER_PEAK} band_k=${W_RICCATI_BOUNDARY_BAND_KAPPA} band_q=${W_RICCATI_BOUNDARY_BAND_Q} shoot=${W_RICCATI_SHOOTING_MATCH} shoot_path=${W_RICCATI_SHOOTING_PATH} ci_local_min=${W_RICCATI_CI_LOCAL_MIN}"
echo "Optim: lr=${LEARNING_RATE} grad_clip=${GRAD_CLIP_NORM} separate_branch=${SEPARATE_BRANCH_OPTIMIZERS} detach_ci_mode=${DETACH_CI_IN_MODE_BRANCH} ci_lr=${CI_BRANCH_LR} mode_lr=${MODE_BRANCH_LR}"
echo "Shoot match: xi_boundary=${RICCATI_SHOOTING_XI_BOUNDARY} start_epoch=${RICCATI_SHOOTING_MATCH_START_EPOCH}"
echo "Shoot path: points=${RICCATI_SHOOTING_PATH_POINTS} xi_boundary=${RICCATI_SHOOTING_PATH_XI_BOUNDARY} start_epoch=${RICCATI_SHOOTING_PATH_START_EPOCH} every=${RICCATI_SHOOTING_PATH_EVERY}"
echo "Initial ci=${INITIAL_CI}"
echo "CI local min: start_epoch=${RICCATI_CI_LOCAL_MIN_START_EPOCH} delta_abs=${RICCATI_CI_LOCAL_MIN_DELTA_ABS} delta_rel=${RICCATI_CI_LOCAL_MIN_DELTA_REL} margin=${RICCATI_CI_LOCAL_MIN_MARGIN}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"
date
nvidia-smi || true

srun bash scripts/run_kh_subsonic_singlecase_pure_physics_scalar_ci_matching_phaseA_ter.sh
