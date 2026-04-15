#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ALPHA_VALUE="${ALPHA_VALUE:-0.5}"
MACH_VALUE="${MACH_VALUE:-0.5}"

python3 scripts/train_kh_subsonic_pinn.py \
  --mach "${MACH_VALUE}" \
  --alpha-min "${ALPHA_VALUE}" \
  --alpha-max "${ALPHA_VALUE}" \
  --epochs 5000 \
  --learning-rate 1e-3 \
  --hidden-dim 160 \
  --mode-depth 4 \
  --ci-depth 2 \
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
  --w-bc-kappa 10.0 \
  --w-bc-q 25.0 \
  --w-ci-supervision 0.0 \
  --w-riccati-anchor 0.0 \
  --w-q-supervision 0.0 \
  --w-riccati-center-kappa 5.0 \
  --w-riccati-center-peak 2.0 \
  --w-riccati-boundary-band-kappa 2.0 \
  --w-riccati-boundary-band-q 8.0 \
  --riccati-center-xi 0.0 \
  --riccati-boundary-band-points 32 \
  --riccati-boundary-band-start 0.94 \
  --riccati-boundary-band-end 0.995 \
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
  --device "${DEVICE:-cpu}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_singlecase_pure_a050_m050_centered}"
