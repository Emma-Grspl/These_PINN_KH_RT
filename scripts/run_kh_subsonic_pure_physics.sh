#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 scripts/train_kh_subsonic_pinn.py \
  --mach 0.5 \
  --alpha-min 0.05 \
  --alpha-max 0.85 \
  --epochs 5000 \
  --learning-rate 1e-3 \
  --hidden-dim 160 \
  --mode-depth 4 \
  --ci-depth 2 \
  --activation tanh \
  --mapping-scale 3.0 \
  --n-interior 512 \
  --n-boundary 64 \
  --n-alpha-supervision 128 \
  --n-anchor-alpha 16 \
  --n-norm-interior 256 \
  --n-reference-alpha 81 \
  --n-audit-alpha 21 \
  --n-mode-audit-alpha 9 \
  --n-mode-audit-y 801 \
  --audit-every 100 \
  --checkpoint-every 500 \
  --focus-fraction 0.0 \
  --focus-half-width 0.03 \
  --neutral-fraction 0.2 \
  --neutral-half-width 0.04 \
  --error-threshold 0.01 \
  --mode-error-threshold 0.12 \
  --max-focus-points 8 \
  --anchor-strategy point \
  --w-pde 1.0 \
  --w-bc-kappa 10.0 \
  --w-bc-q 20.0 \
  --w-ci-supervision 0.0 \
  --w-riccati-anchor 0.0 \
  --w-q-supervision 0.0 \
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
  --mode-experts 2 \
  --alpha-split-threshold 0.40 \
  --device "${DEVICE:-cpu}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_riccati_pure_physics}"
