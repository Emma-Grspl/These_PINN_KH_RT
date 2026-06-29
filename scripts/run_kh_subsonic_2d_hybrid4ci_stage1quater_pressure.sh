#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAGE0_CHECKPOINT="${STAGE0_CHECKPOINT:-model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_subsonic_2d_hybrid4ci_stage1quater_pressure}"
EPOCHS="${EPOCHS:-1500}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MACH_VALUES="${MACH_VALUES:-0.1 0.3 0.5 0.7}"
ALPHA_MIN="${ALPHA_MIN:-0.1}"
ALPHA_MAX="${ALPHA_MAX:-0.8}"
ANCHOR_ALPHAS="${ANCHOR_ALPHAS:-0.1 0.3 0.55 0.8}"
N_INTERIOR="${N_INTERIOR:-256}"
N_BOUNDARY="${N_BOUNDARY:-64}"
N_CENTER="${N_CENTER:-192}"
N_SYM="${N_SYM:-128}"
SYM_YMAX="${SYM_YMAX:-15.0}"
N_ALPHA_SAMPLES="${N_ALPHA_SAMPLES:-8}"
N_MACH_SAMPLES="${N_MACH_SAMPLES:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
DEPTH="${DEPTH:-4}"
ACTIVATION="${ACTIVATION:-tanh}"
FREEZE_CI="${FREEZE_CI:-1}"
DETACH_CI_IN_MODE_BRANCH="${DETACH_CI_IN_MODE_BRANCH:-1}"
W_PDE="${W_PDE:-1.0}"
W_BC="${W_BC:-20.0}"
W_GAUGE="${W_GAUGE:-100.0}"
W_CENTER_PDE="${W_CENTER_PDE:-1.0}"
W_SYM="${W_SYM:-0.0}"
W_CENTER_DERIV="${W_CENTER_DERIV:-0.0}"
W_CI_ANCHOR="${W_CI_ANCHOR:-1.0}"
YMAX="${YMAX:-75.0}"
ENVELOPE_EPS="${ENVELOPE_EPS:-1.0}"
CENTER_WIDTH="${CENTER_WIDTH:-3.0}"
CENTER_FRACTION="${CENTER_FRACTION:-0.6}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
AUDIT_EVERY="${AUDIT_EVERY:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
BEST_METRIC="${BEST_METRIC:-loss_total}"
REFERENCE_CACHE="${REFERENCE_CACHE:-}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-1234}"

FREEZE_ARGS=(--freeze-ci)
if [[ "${FREEZE_CI}" == "0" ]]; then
  FREEZE_ARGS=(--no-freeze-ci)
fi

DETACH_ARGS=(--detach-ci-in-mode-branch)
if [[ "${DETACH_CI_IN_MODE_BRANCH}" == "0" ]]; then
  DETACH_ARGS=(--no-detach-ci-in-mode-branch)
fi

CMD=(
  python3 scripts/train_kh_subsonic_2d_hybrid4ci_stage1quater_pressure.py
  --stage0-checkpoint "${STAGE0_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --mach-values ${MACH_VALUES}
  --alpha-min "${ALPHA_MIN}"
  --alpha-max "${ALPHA_MAX}"
  --anchor-alphas ${ANCHOR_ALPHAS}
  --n-interior "${N_INTERIOR}"
  --n-boundary "${N_BOUNDARY}"
  --n-center "${N_CENTER}"
  --n-sym "${N_SYM}"
  --sym-ymax "${SYM_YMAX}"
  --n-alpha-samples "${N_ALPHA_SAMPLES}"
  --n-mach-samples "${N_MACH_SAMPLES}"
  --hidden-dim "${HIDDEN_DIM}"
  --depth "${DEPTH}"
  --activation "${ACTIVATION}"
  "${FREEZE_ARGS[@]}"
  "${DETACH_ARGS[@]}"
  --w-pde "${W_PDE}"
  --w-bc "${W_BC}"
  --w-gauge "${W_GAUGE}"
  --w-center-pde "${W_CENTER_PDE}"
  --w-sym "${W_SYM}"
  --w-center-deriv "${W_CENTER_DERIV}"
  --w-ci-anchor "${W_CI_ANCHOR}"
  --ymax "${YMAX}"
  --envelope-eps "${ENVELOPE_EPS}"
  --center-width "${CENTER_WIDTH}"
  --center-fraction "${CENTER_FRACTION}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --audit-every "${AUDIT_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --best-metric "${BEST_METRIC}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ -n "${REFERENCE_CACHE}" ]]; then
  CMD+=(--reference-cache "${REFERENCE_CACHE}")
fi

echo "Running Stage 1quater pressure-first PINN"
echo "stage0_checkpoint=${STAGE0_CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "mach_values=${MACH_VALUES}"
echo "alpha_range=[${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "epochs=${EPOCHS} lr=${LEARNING_RATE}"
echo "weights: pde=${W_PDE} bc=${W_BC} gauge=${W_GAUGE} center_pde=${W_CENTER_PDE} sym=${W_SYM} center_deriv=${W_CENTER_DERIV}"
echo "symmetry: n_sym=${N_SYM} sym_ymax=${SYM_YMAX}"
echo "ymax=${YMAX} envelope_eps=${ENVELOPE_EPS}"
echo "freeze_ci=${FREEZE_CI} detach_ci_in_mode_branch=${DETACH_CI_IN_MODE_BRANCH}"

"${CMD[@]}"
