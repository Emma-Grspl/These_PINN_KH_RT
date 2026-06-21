#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${MACH_VALUE:=0.5}"
: "${ALPHA_MIN:=0.10}"
: "${ALPHA_MAX:=0.80}"
: "${EPOCHS:=1000}"
: "${LEARNING_RATE:=1e-2}"
: "${HIDDEN_DIM:=160}"
: "${CI_DEPTH:=2}"
: "${INITIAL_CI:=0.2}"
: "${DEVICE:=cpu}"
: "${W_STAGE0_ANCHOR:=100.0}"
: "${W_STAGE0_MONOTONE:=1.0}"
: "${W_STAGE0_SMOOTH:=0.0}"
: "${STAGE0_LBFGS_STEPS:=500}"
: "${STAGE0_LBFGS_LR:=0.5}"
: "${STAGE0_N_SHAPE_GRID:=129}"
: "${STAGE0_TARGET_MAX_ABS:=1e-3}"
: "${AUDIT_EVERY:=100}"
: "${CI_SUP_FIXED_ALPHAS:?CI_SUP_FIXED_ALPHAS must contain the alpha anchors}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

read -r -a CI_ALPHA_ARRAY <<< "${CI_SUP_FIXED_ALPHAS}"
if [[ -n "${CI_SUP_COUNT:-}" && "${#CI_ALPHA_ARRAY[@]}" -ne "${CI_SUP_COUNT}" ]]; then
  echo "Mismatch: CI_SUP_COUNT=${CI_SUP_COUNT} but got ${#CI_ALPHA_ARRAY[@]} fixed alphas." >&2
  exit 1
fi

args=(
  --mach "${MACH_VALUE}"
  --alpha-min "${ALPHA_MIN}"
  --alpha-max "${ALPHA_MAX}"
  --epochs "${EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --hidden-dim "${HIDDEN_DIM}"
  --ci-depth "${CI_DEPTH}"
  --initial-ci "${INITIAL_CI}"
  --device "${DEVICE}"
  --w-anchor "${W_STAGE0_ANCHOR}"
  --w-monotone "${W_STAGE0_MONOTONE}"
  --w-smooth "${W_STAGE0_SMOOTH}"
  --lbfgs-steps "${STAGE0_LBFGS_STEPS}"
  --lbfgs-lr "${STAGE0_LBFGS_LR}"
  --n-shape-grid "${STAGE0_N_SHAPE_GRID}"
  --target-max-abs "${STAGE0_TARGET_MAX_ABS}"
  --audit-every "${AUDIT_EVERY}"
  --output-dir "${OUTPUT_DIR}"
  --anchors
)

for alpha in "${CI_ALPHA_ARRAY[@]}"; do
  args+=("${alpha}")
done

if [[ -n "${REFERENCE_CSV:-}" ]]; then
  args+=(--reference-csv "${REFERENCE_CSV}")
fi

if [[ "${STAGE0_NO_FAIL_ON_TARGET:-0}" == "1" ]]; then
  args+=(--no-fail-on-target)
fi

python3 scripts/train_kh_subsonic_ci_stage0_anchor_lock.py "${args[@]}"
