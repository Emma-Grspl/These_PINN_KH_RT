#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${STAGE0_CHECKPOINT:=model_saved/kh_subsonic_2d_hybrid4ci_stage0_anchor_lock_no_monotone_lbfgs/model_best.pt}"
: "${MACH_VALUES:=0.1 0.3 0.5 0.7}"
: "${ALPHA_MIN:=0.10}"
: "${ALPHA_MAX:=0.80}"
: "${ANCHOR_ALPHAS:=0.10 0.30 0.55 0.80}"
: "${N_INTERIOR:=512}"
: "${N_BOUNDARY:=96}"
: "${N_CENTER:=256}"
: "${CENTER_WIDTH:=2.0}"
: "${CENTER_FRACTION:=0.5}"
: "${N_ALPHA_SAMPLES:=12}"
: "${N_MACH_SAMPLES:=4}"
: "${EPOCHS:=8000}"
: "${LEARNING_RATE:=5e-5}"
: "${GRAD_CLIP_NORM:=1.0}"
: "${DEVICE:=cpu}"
: "${SEED:=1234}"
: "${FREEZE_CI:=1}"
: "${DETACH_CI_IN_MODE_BRANCH:=1}"
: "${AUDIT_EVERY:=100}"
: "${CHECKPOINT_EVERY:=500}"
: "${W_PDE:=1.0}"
: "${W_BC_KAPPA:=20.0}"
: "${W_BC_Q:=60.0}"
: "${W_MATCH:=1.0}"
: "${W_CENTER_PDE:=1.0}"
: "${W_NORM:=0.0}"
: "${W_PHASE:=0.0}"
: "${W_CI_ANCHOR:=1.0}"
: "${MATCH_Y_VALUES:=-1.0 -0.5 0.0 0.5 1.0}"
: "${SHOOT_YMAX:=40.0}"
: "${SHOOT_STEPS:=512}"
: "${MATCH_WARMUP_EPOCHS:=1000}"
: "${BEST_METRIC:=loss_total}"
: "${OUTPUT_DIR:=model_saved/kh_subsonic_2d_hybrid4ci_stage1ter_riccati_matching}"

read -r -a MACH_ARRAY <<< "${MACH_VALUES}"
read -r -a ANCHOR_ARRAY <<< "${ANCHOR_ALPHAS}"
read -r -a MATCH_Y_ARRAY <<< "${MATCH_Y_VALUES}"

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
  --n-center "${N_CENTER}"
  --center-width "${CENTER_WIDTH}"
  --center-fraction "${CENTER_FRACTION}"
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
  --w-match "${W_MATCH}"
  --w-center-pde "${W_CENTER_PDE}"
  --w-norm "${W_NORM}"
  --w-phase "${W_PHASE}"
  --w-ci-anchor "${W_CI_ANCHOR}"
  --shoot-ymax "${SHOOT_YMAX}"
  --shoot-steps "${SHOOT_STEPS}"
  --match-warmup-epochs "${MATCH_WARMUP_EPOCHS}"
  --best-metric "${BEST_METRIC}"
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

args+=(--match-y-values)
for y_value in "${MATCH_Y_ARRAY[@]}"; do
  args+=("${y_value}")
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

python3 scripts/train_kh_subsonic_2d_hybrid4ci_stage1ter_matching.py "${args[@]}"
