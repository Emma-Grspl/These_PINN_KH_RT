#!/usr/bin/env bash
set -euo pipefail

python3 scripts/search_kh_subsonic_singlecase_external_ci.py \
  --mach "${MACH_VALUE:-0.5}" \
  --alpha "${ALPHA_VALUE:-0.5}" \
  --ci-min "${CI_MIN:-0.02}" \
  --ci-max "${CI_MAX:-0.22}" \
  --ci-count "${CI_COUNT:-9}" \
  --epochs "${EPOCHS:-2500}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_singlecase_external_ci_search_a050_m050}"
