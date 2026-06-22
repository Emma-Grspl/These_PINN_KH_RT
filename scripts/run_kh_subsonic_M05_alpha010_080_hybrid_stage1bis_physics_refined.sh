#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${MACH_VALUE:=0.5}"
: "${ALPHA_MIN:=0.10}"
: "${ALPHA_MAX:=0.80}"
: "${EPOCHS:=6000}"
: "${LEARNING_RATE:=2e-4}"
: "${MODE_BRANCH_LR:=2e-4}"
: "${GRAD_CLIP_NORM:=0.75}"
: "${HIDDEN_DIM:=160}"
: "${MODE_EXPERTS:=2}"
: "${ALPHA_SPLIT_THRESHOLD:=0.40}"
: "${DEVICE:=cuda}"
: "${CI_SUP_COUNT:?CI_SUP_COUNT must be set}"
: "${CI_SUP_FIXED_ALPHAS:?CI_SUP_FIXED_ALPHAS must be set}"
: "${STAGE1_MODEL_PATH:?STAGE1_MODEL_PATH must point to a Stage 1 model_best.pt}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

if [[ ! -f "${STAGE1_MODEL_PATH}" ]]; then
  echo "Stage 1 checkpoint not found: ${STAGE1_MODEL_PATH}" >&2
  exit 1
fi

read -r -a CI_ALPHA_ARRAY <<< "${CI_SUP_FIXED_ALPHAS}"
if [[ "${#CI_ALPHA_ARRAY[@]}" -ne "${CI_SUP_COUNT}" ]]; then
  echo "Mismatch: CI_SUP_COUNT=${CI_SUP_COUNT} but got ${#CI_ALPHA_ARRAY[@]} fixed alphas." >&2
  exit 1
fi

: "${N_INTERIOR:=2048}"
: "${N_BOUNDARY:=128}"
: "${N_NORM_INTERIOR:=512}"
: "${W_PDE:=1.5}"
: "${W_BC_KAPPA:=15.0}"
: "${W_BC_Q:=35.0}"
: "${W_RICCATI_CENTER_KAPPA:=1.0}"
: "${W_RICCATI_CENTER_PEAK:=0.0}"
: "${W_RICCATI_BOUNDARY_BAND_KAPPA:=1.0}"
: "${W_RICCATI_BOUNDARY_BAND_Q:=5.0}"
: "${RICCATI_BOUNDARY_BAND_POINTS:=64}"
: "${RICCATI_BOUNDARY_BAND_START:=0.88}"
: "${RICCATI_BOUNDARY_BAND_END:=0.995}"
: "${W_RICCATI_SHOOTING_MATCH:=2.0}"
: "${RICCATI_SHOOTING_MATCH_START_EPOCH:=500}"
: "${RICCATI_SHOOTING_STEPS:=384}"
: "${RICCATI_SHOOTING_XI_BOUNDARY:=0.985}"
: "${W_RICCATI_SHOOTING_PATH:=1.0}"
: "${RICCATI_SHOOTING_PATH_POINTS:=33}"
: "${RICCATI_SHOOTING_PATH_XI_BOUNDARY:=0.90}"
: "${RICCATI_SHOOTING_PATH_START_EPOCH:=1000}"
: "${RICCATI_SHOOTING_PATH_EVERY:=10}"
: "${MODE_CENTER_FRACTION:=0.65}"
: "${MODE_CENTER_HALF_WIDTH:=0.45}"
: "${AUDIT_EVERY:=100}"
: "${CHECKPOINT_EVERY:=500}"

python3 scripts/train_kh_subsonic_pinn.py \
  --mach "${MACH_VALUE}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --mode-depth 4 \
  --ci-depth 2 \
  --activation tanh \
  --mapping-scale 3.0 \
  --n-interior "${N_INTERIOR}" \
  --n-boundary "${N_BOUNDARY}" \
  --n-alpha-supervision "${CI_SUP_COUNT}" \
  --ci-supervision-fixed-alphas "${CI_ALPHA_ARRAY[@]}" \
  --n-anchor-alpha 32 \
  --n-norm-interior "${N_NORM_INTERIOR}" \
  --n-reference-alpha 121 \
  --n-audit-alpha 31 \
  --n-mode-audit-alpha 11 \
  --n-mode-audit-y 1001 \
  --audit-every "${AUDIT_EVERY}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --mode-representation riccati \
  --mode-experts "${MODE_EXPERTS}" \
  --alpha-split-threshold "${ALPHA_SPLIT_THRESHOLD}" \
  --freeze-ci \
  --disable-classic-ci-supervision \
  --separate-branch-optimizers \
  --detach-ci-in-mode-branch \
  --mode-branch-lr "${MODE_BRANCH_LR}" \
  --initial-model-path "${STAGE1_MODEL_PATH}" \
  --no-initial-model-strict \
  --focus-fraction 0.0 \
  --neutral-fraction 0.0 \
  --anchor-strategy band \
  --anchor-half-width 0.12 \
  --mode-center-fraction "${MODE_CENTER_FRACTION}" \
  --mode-center-half-width "${MODE_CENTER_HALF_WIDTH}" \
  --w-pde "${W_PDE}" \
  --w-bc-kappa "${W_BC_KAPPA}" \
  --w-bc-q "${W_BC_Q}" \
  --w-ci-supervision 0.0 \
  --w-ci-stability-outside 0.0 \
  --w-ci-neutrality 0.0 \
  --w-ci-low-alpha-zero 0.0 \
  --w-ci-smoothness 0.0 \
  --w-riccati-anchor 0.0 \
  --w-q-supervision 0.0 \
  --w-riccati-center-kappa "${W_RICCATI_CENTER_KAPPA}" \
  --w-riccati-center-peak "${W_RICCATI_CENTER_PEAK}" \
  --w-riccati-boundary-band-kappa "${W_RICCATI_BOUNDARY_BAND_KAPPA}" \
  --w-riccati-boundary-band-q "${W_RICCATI_BOUNDARY_BAND_Q}" \
  --riccati-boundary-band-points "${RICCATI_BOUNDARY_BAND_POINTS}" \
  --riccati-boundary-band-start "${RICCATI_BOUNDARY_BAND_START}" \
  --riccati-boundary-band-end "${RICCATI_BOUNDARY_BAND_END}" \
  --w-riccati-shooting-match "${W_RICCATI_SHOOTING_MATCH}" \
  --riccati-shooting-match-start-epoch "${RICCATI_SHOOTING_MATCH_START_EPOCH}" \
  --riccati-shooting-steps "${RICCATI_SHOOTING_STEPS}" \
  --riccati-shooting-xi-boundary "${RICCATI_SHOOTING_XI_BOUNDARY}" \
  --w-riccati-shooting-path "${W_RICCATI_SHOOTING_PATH}" \
  --riccati-shooting-path-points "${RICCATI_SHOOTING_PATH_POINTS}" \
  --riccati-shooting-path-xi-boundary "${RICCATI_SHOOTING_PATH_XI_BOUNDARY}" \
  --riccati-shooting-path-start-epoch "${RICCATI_SHOOTING_PATH_START_EPOCH}" \
  --riccati-shooting-path-every "${RICCATI_SHOOTING_PATH_EVERY}" \
  --w-riccati-ci-local-min 0.0 \
  --mode-low-alpha-threshold 0.25 \
  --mode-low-alpha-weight 2.0 \
  --mode-low-alpha-audit-fraction 0.65 \
  --audit-ci-weight 10.0 \
  --audit-env-weight 1.0 \
  --audit-phase-weight 0.5 \
  --audit-peak-weight 0.25 \
  --phase-mask-fraction 0.15 \
  --classic-n-points 561 \
  --classic-mapping-scale 3.0 \
  --classic-xi-max 0.99 \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}"
