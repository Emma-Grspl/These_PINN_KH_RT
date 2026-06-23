#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${STAGE0_CHECKPOINT:=model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt}"
: "${MACH_VALUES:=0.1 0.3 0.5 0.7}"
: "${ALPHA_MIN:=0.10}"
: "${ALPHA_MAX:=0.80}"
: "${ANCHOR_ALPHAS:=0.10 0.30 0.55 0.80}"
: "${N_INTERIOR:=128}"
: "${N_BOUNDARY:=32}"
: "${N_ALPHA_SAMPLES:=8}"
: "${N_MACH_SAMPLES:=4}"
: "${EPOCHS:=5000}"
: "${LEARNING_RATE:=1e-4}"
: "${GRAD_CLIP_NORM:=1.0}"
: "${DEVICE:=cpu}"
: "${SEED:=1234}"
: "${FREEZE_CI:=1}"
: "${DETACH_CI_IN_MODE_BRANCH:=1}"
: "${AUDIT_EVERY:=100}"
: "${CHECKPOINT_EVERY:=500}"
: "${W_PDE:=1.0}"
: "${W_BC_KAPPA:=10.0}"
: "${W_BC_Q:=25.0}"
: "${W_NORM:=1.0}"
: "${W_PHASE:=1.0}"
: "${W_SHOOTING:=0.0}"
: "${W_CI_ANCHOR:=1.0}"
: "${OUTPUT_DIR:=model_saved/kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0}"

read -r -a MACH_ARRAY <<< "${MACH_VALUES}"
read -r -a ANCHOR_ARRAY <<< "${ANCHOR_ALPHAS}"

if [[ ! -f "${STAGE0_CHECKPOINT}" ]]; then
  echo "Stage 0 checkpoint not found: ${STAGE0_CHECKPOINT}" >&2
  exit 1
fi

args=(
  --stage0-checkpoint "${STAGE0_CHECKPOINT}"
  --alpha-min "${ALPHA_MIN}"
  --alpha-max "${ALPHA_MAX}"
  --n-interior "${N_INTERIOR}"
  --n-boundary "${N_BOUNDARY}"
  --n-alpha-samples "${N_ALPHA_SAMPLES}"
  --n-mach-samples "${N_MACH_SAMPLES}"
  --epochs "${EPOCHS}"
  --lr "${LEARNING_RATE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --audit-every "${AUDIT_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --w-pde "${W_PDE}"
  --w-bc-kappa "${W_BC_KAPPA}"
  --w-bc-q "${W_BC_Q}"
  --w-norm "${W_NORM}"
  --w-phase "${W_PHASE}"
  --w-shooting "${W_SHOOTING}"
  --w-ci-anchor "${W_CI_ANCHOR}"
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

if [[ "${FREEZE_CI}" == "1" ]]; then
  args+=(--freeze-ci)
else
  args+=(--no-freeze-ci)
fi

if [[ "${DETACH_CI_IN_MODE_BRANCH}" == "1" ]]; then
  args+=(--detach-ci-in-mode-branch)
else
  args+=(--no-detach-ci-in-mode-branch)
fi

python3 scripts/train_kh_subsonic_2d_hybrid4ci_stage1_mode_from_ci_stage0.py "${args[@]}"
