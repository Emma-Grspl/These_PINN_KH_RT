#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ALPHA_VALUE="${ALPHA_VALUE:-0.5}"
MACH_VALUE="${MACH_VALUE:-0.5}"
ALPHA_TAG="$(printf '%0.3f' "${ALPHA_VALUE}" | tr -d '.')"
MACH_TAG="$(printf '%0.3f' "${MACH_VALUE}" | tr -d '.')"

W_BC_KAPPA="${W_BC_KAPPA:-10.0}"
W_BC_Q="${W_BC_Q:-25.0}"
W_RICCATI_CENTER_KAPPA="${W_RICCATI_CENTER_KAPPA:-1.0}"
W_RICCATI_CENTER_PEAK="${W_RICCATI_CENTER_PEAK:-0.0}"
W_RICCATI_BOUNDARY_BAND_KAPPA="${W_RICCATI_BOUNDARY_BAND_KAPPA:-0.5}"
W_RICCATI_BOUNDARY_BAND_Q="${W_RICCATI_BOUNDARY_BAND_Q:-2.0}"
W_RICCATI_SHOOTING_MATCH="${W_RICCATI_SHOOTING_MATCH:-50.0}"
W_RICCATI_SHOOTING_PATH="${W_RICCATI_SHOOTING_PATH:-0.0}"
W_RICCATI_CI_LOCAL_MIN="${W_RICCATI_CI_LOCAL_MIN:-20.0}"
RICCATI_SHOOTING_PATH_POINTS="${RICCATI_SHOOTING_PATH_POINTS:-33}"
RICCATI_SHOOTING_XI_BOUNDARY="${RICCATI_SHOOTING_XI_BOUNDARY:-0.995}"
RICCATI_SHOOTING_PATH_XI_BOUNDARY="${RICCATI_SHOOTING_PATH_XI_BOUNDARY:-0.94}"
RICCATI_SHOOTING_PATH_START_EPOCH="${RICCATI_SHOOTING_PATH_START_EPOCH:-0}"
RICCATI_SHOOTING_PATH_EVERY="${RICCATI_SHOOTING_PATH_EVERY:-1}"
RICCATI_CI_LOCAL_MIN_DELTA_ABS="${RICCATI_CI_LOCAL_MIN_DELTA_ABS:-0.002}"
RICCATI_CI_LOCAL_MIN_DELTA_REL="${RICCATI_CI_LOCAL_MIN_DELTA_REL:-0.02}"
RICCATI_CI_LOCAL_MIN_MARGIN="${RICCATI_CI_LOCAL_MIN_MARGIN:-1e-4}"
INITIAL_CI="${INITIAL_CI:-0.2}"
EPOCHS="${EPOCHS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
SEPARATE_BRANCH_OPTIMIZERS="${SEPARATE_BRANCH_OPTIMIZERS:-1}"
DETACH_CI_IN_MODE_BRANCH="${DETACH_CI_IN_MODE_BRANCH:-1}"
CI_BRANCH_LR="${CI_BRANCH_LR:-5e-5}"
MODE_BRANCH_LR="${MODE_BRANCH_LR:-2e-4}"
RICCATI_SHOOTING_MATCH_START_EPOCH="${RICCATI_SHOOTING_MATCH_START_EPOCH:-200}"
RICCATI_CI_LOCAL_MIN_START_EPOCH="${RICCATI_CI_LOCAL_MIN_START_EPOCH:-900}"

python3 scripts/train_kh_subsonic_pinn.py \
  --mach "${MACH_VALUE}" \
  --alpha-min "${ALPHA_VALUE}" \
  --alpha-max "${ALPHA_VALUE}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --hidden-dim 160 \
  --mode-depth 4 \
  --ci-depth 2 \
  --fixed-scalar-ci \
  --initial-ci "${INITIAL_CI}" \
  --activation tanh \
  --mapping-scale 3.0 \
  --n-interior 512 \
  --n-boundary 96 \
  --n-alpha-supervision 64 \
  --n-anchor-alpha 8 \
  --n-norm-interior 256 \
  --n-reference-alpha 1 \
  --n-audit-alpha 1 \
  --n-mode-audit-alpha 1 \
  --n-mode-audit-y 1201 \
  --audit-every 100 \
  --checkpoint-every 500 \
  --focus-fraction 0.0 \
  --focus-half-width 0.0 \
  --neutral-fraction 0.0 \
  --error-threshold 0.0 \
  --mode-error-threshold 0.0 \
  --max-focus-points 0 \
  --anchor-strategy point \
  --anchor-half-width 0.10 \
  --w-pde 1.0 \
  --w-bc-kappa "${W_BC_KAPPA}" \
  --w-bc-q "${W_BC_Q}" \
  --w-ci-supervision 0.0 \
  --w-riccati-anchor 0.0 \
  --w-q-supervision 0.0 \
  --w-riccati-center-kappa "${W_RICCATI_CENTER_KAPPA}" \
  --w-riccati-center-peak "${W_RICCATI_CENTER_PEAK}" \
  --w-riccati-boundary-band-kappa "${W_RICCATI_BOUNDARY_BAND_KAPPA}" \
  --w-riccati-boundary-band-q "${W_RICCATI_BOUNDARY_BAND_Q}" \
  --w-riccati-shooting-match "${W_RICCATI_SHOOTING_MATCH}" \
  --w-riccati-shooting-path "${W_RICCATI_SHOOTING_PATH}" \
  --w-riccati-ci-local-min "${W_RICCATI_CI_LOCAL_MIN}" \
  --riccati-shooting-match-start-epoch "${RICCATI_SHOOTING_MATCH_START_EPOCH}" \
  --riccati-center-xi 0.0 \
  --riccati-boundary-band-points 32 \
  --riccati-boundary-band-start 0.94 \
  --riccati-boundary-band-end 0.995 \
  --riccati-shooting-steps 512 \
  --riccati-shooting-xi-boundary "${RICCATI_SHOOTING_XI_BOUNDARY}" \
  --riccati-shooting-path-points "${RICCATI_SHOOTING_PATH_POINTS}" \
  --riccati-shooting-path-xi-boundary "${RICCATI_SHOOTING_PATH_XI_BOUNDARY}" \
  --riccati-shooting-path-start-epoch "${RICCATI_SHOOTING_PATH_START_EPOCH}" \
  --riccati-shooting-path-every "${RICCATI_SHOOTING_PATH_EVERY}" \
  --riccati-ci-local-min-start-epoch "${RICCATI_CI_LOCAL_MIN_START_EPOCH}" \
  --riccati-ci-local-min-delta-abs "${RICCATI_CI_LOCAL_MIN_DELTA_ABS}" \
  --riccati-ci-local-min-delta-rel "${RICCATI_CI_LOCAL_MIN_DELTA_REL}" \
  --riccati-ci-local-min-margin "${RICCATI_CI_LOCAL_MIN_MARGIN}" \
  --disable-classic-ci-supervision \
  --audit-ci-weight 10.0 \
  --audit-env-weight 1.0 \
  --audit-phase-weight 0.5 \
  --audit-peak-weight 0.25 \
  --phase-mask-fraction 0.15 \
  --classic-n-points 561 \
  --classic-mapping-scale 3.0 \
  --classic-xi-max 0.99 \
  --mode-representation riccati \
  --mode-experts 1 \
  $( [[ "${SEPARATE_BRANCH_OPTIMIZERS}" == "1" ]] && printf '%s' "--separate-branch-optimizers" ) \
  $( [[ "${DETACH_CI_IN_MODE_BRANCH}" == "1" ]] && printf '%s' "--detach-ci-in-mode-branch" ) \
  --ci-branch-lr "${CI_BRANCH_LR}" \
  --mode-branch-lr "${MODE_BRANCH_LR}" \
  --device "${DEVICE:-cpu}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_singlecase_phaseA_bis_a${ALPHA_TAG}_m${MACH_TAG}_scalar_ci_matching}"
