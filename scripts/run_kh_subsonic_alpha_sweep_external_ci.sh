#!/usr/bin/env bash
set -euo pipefail

python3 scripts/search_kh_subsonic_alpha_sweep_external_ci.py \
  --mach "${MACH_VALUE:-0.5}" \
  --alpha-min "${ALPHA_MIN:-0.2}" \
  --alpha-max "${ALPHA_MAX:-0.8}" \
  --alpha-count "${ALPHA_COUNT:-5}" \
  --ci-min "${CI_MIN:-0.2}" \
  --ci-max "${CI_MAX:-0.34}" \
  --ci-count "${CI_COUNT:-8}" \
  --epochs "${EPOCHS:-1500}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_external_ci_alpha_sweep}"
