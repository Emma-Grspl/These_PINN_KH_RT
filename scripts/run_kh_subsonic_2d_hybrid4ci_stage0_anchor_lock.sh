#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${MACH_VALUES:=0.1 0.3 0.5 0.7}"
: "${ALPHA_MIN:=0.10}"
: "${ALPHA_MAX:=0.80}"
: "${ANCHOR_ALPHAS:=0.10 0.30 0.55 0.80}"
: "${EPOCHS:=2000}"
: "${LEARNING_RATE:=1e-3}"
: "${HIDDEN_DIM:=160}"
: "${CI_DEPTH:=2}"
: "${INITIAL_CI:=0.2}"
: "${DEVICE:=cpu}"
: "${SEED:=1234}"
: "${W_STAGE0_ANCHOR:=100.0}"
: "${W_STAGE0_MONOTONE_ALPHA:=1.0}"
: "${W_STAGE0_SMOOTH_ALPHA:=0.0}"
: "${W_STAGE0_SMOOTH_MACH:=0.0}"
: "${N_SHAPE_ALPHA:=65}"
: "${N_SHAPE_MACH:=25}"
: "${AUDIT_EVERY:=50}"
: "${STAGE0_LBFGS_STEPS:=0}"
: "${STAGE0_LBFGS_LR:=0.5}"
: "${STAGE0_TARGET_MAX_ABS:=1e-3}"
: "${STAGE0_TARGET_MAX_REL:=5e-2}"
: "${OUTPUT_DIR:=model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock}"

read -r -a MACH_ARRAY <<< "${MACH_VALUES}"
read -r -a ANCHOR_ARRAY <<< "${ANCHOR_ALPHAS}"

args=(
  --alpha-min "${ALPHA_MIN}"
  --alpha-max "${ALPHA_MAX}"
  --epochs "${EPOCHS}"
  --lr "${LEARNING_RATE}"
  --hidden-dim "${HIDDEN_DIM}"
  --ci-depth "${CI_DEPTH}"
  --initial-ci "${INITIAL_CI}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --w-anchor "${W_STAGE0_ANCHOR}"
  --w-monotone-alpha "${W_STAGE0_MONOTONE_ALPHA}"
  --w-smooth-alpha "${W_STAGE0_SMOOTH_ALPHA}"
  --w-smooth-mach "${W_STAGE0_SMOOTH_MACH}"
  --n-shape-alpha "${N_SHAPE_ALPHA}"
  --n-shape-mach "${N_SHAPE_MACH}"
  --audit-every "${AUDIT_EVERY}"
  --lbfgs-steps "${STAGE0_LBFGS_STEPS}"
  --lbfgs-lr "${STAGE0_LBFGS_LR}"
  --target-max-abs "${STAGE0_TARGET_MAX_ABS}"
  --target-max-rel "${STAGE0_TARGET_MAX_REL}"
  --output-dir "${OUTPUT_DIR}"
  --mach-values
)

for mach in "${MACH_ARRAY[@]}"; do
  args+=("${mach}")
done

args+=(--anchor-alphas)
for alpha in "${ANCHOR_ARRAY[@]}"; do
  args+=("${alpha}")
done

if [[ -n "${REFERENCE_CACHE:-}" ]]; then
  args+=(--reference-cache "${REFERENCE_CACHE}")
fi

if [[ "${FAIL_ON_TARGET:-0}" == "1" ]]; then
  args+=(--fail-on-target)
fi

python3 scripts/train_kh_subsonic_2d_ci_stage0_anchor_lock.py "${args[@]}"
