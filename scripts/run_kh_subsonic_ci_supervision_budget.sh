#!/usr/bin/env bash
set -euo pipefail

python3 scripts/ablate_kh_subsonic_ci_supervision_budget.py \
  --mach "${MACH_VALUE:-0.5}" \
  --alpha-min "${ALPHA_MIN:-0.05}" \
  --alpha-max "${ALPHA_MAX:-0.85}" \
  --epochs "${EPOCHS:-3000}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --counts "${COUNTS:-4,8,16,32,64,128}" \
  --ci-mae-target "${CI_MAE_TARGET:-0.02}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_budget}"
