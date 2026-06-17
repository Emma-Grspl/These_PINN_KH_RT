#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MACH_VALUE="${MACH_VALUE:-0.5}"
ALPHA_MIN="${ALPHA_MIN:-0.10}"
ALPHA_MAX="${ALPHA_MAX:-0.80}"
EPOCHS="${EPOCHS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"

CI_SUP_COUNT="${CI_SUP_COUNT:?CI_SUP_COUNT must be set}"
CI_SUP_FIXED_ALPHAS="${CI_SUP_FIXED_ALPHAS:?CI_SUP_FIXED_ALPHAS must be set}"
W_CI_SUPERVISION="${W_CI_SUPERVISION:-5.0}"

read -r -a CI_ALPHA_ARRAY <<< "${CI_SUP_FIXED_ALPHAS}"
if [[ "${#CI_ALPHA_ARRAY[@]}" -ne "${CI_SUP_COUNT}" ]]; then
  echo "Mismatch: CI_SUP_COUNT=${CI_SUP_COUNT} but got ${#CI_ALPHA_ARRAY[@]} fixed alphas." >&2
  exit 1
fi

W_BC_KAPPA="${W_BC_KAPPA:-10.0}"
W_BC_Q="${W_BC_Q:-20.0}"
W_RICCATI_CENTER_KAPPA="${W_RICCATI_CENTER_KAPPA:-5.0}"
W_RICCATI_CENTER_PEAK="${W_RICCATI_CENTER_PEAK:-2.0}"
W_RICCATI_BOUNDARY_BAND_KAPPA="${W_RICCATI_BOUNDARY_BAND_KAPPA:-2.0}"
W_RICCATI_BOUNDARY_BAND_Q="${W_RICCATI_BOUNDARY_BAND_Q:-8.0}"
W_CI_LOW_ALPHA_ZERO="${W_CI_LOW_ALPHA_ZERO:-10.0}"
W_CI_SMOOTHNESS="${W_CI_SMOOTHNESS:-0.5}"
N_CI_SPECTRAL_GRID="${N_CI_SPECTRAL_GRID:-129}"

python3 scripts/train_kh_subsonic_pinn.py \
  --mach "${MACH_VALUE}" \
  --alpha-min "${ALPHA_MIN}" \
  --alpha-max "${ALPHA_MAX}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --hidden-dim 160 \
  --mode-depth 4 \
  --ci-depth 2 \
  --activation tanh \
  --mapping-scale 3.0 \
  --n-interior 512 \
  --n-boundary 64 \
  --n-alpha-supervision "${CI_SUP_COUNT}" \
  --ci-supervision-fixed-alphas "${CI_ALPHA_ARRAY[@]}" \
  --n-anchor-alpha 32 \
  --n-norm-interior 256 \
  --n-reference-alpha 121 \
  --n-audit-alpha 31 \
  --n-mode-audit-alpha 11 \
  --n-mode-audit-y 801 \
  --audit-every 100 \
  --checkpoint-every 500 \
  --mode-representation riccati \
  --mode-experts 2 \
  --alpha-split-threshold 0.40 \
  --focus-fraction 0.0 \
  --neutral-fraction 0.0 \
  --anchor-strategy band \
  --anchor-half-width 0.12 \
  --mode-center-fraction 1.0 \
  --mode-center-half-width 0.30 \
  --w-pde 1.0 \
  --w-bc-kappa "${W_BC_KAPPA}" \
  --w-bc-q "${W_BC_Q}" \
  --w-ci-supervision "${W_CI_SUPERVISION}" \
  --w-riccati-anchor 0.0 \
  --w-q-supervision 0.0 \
  --w-riccati-center-kappa "${W_RICCATI_CENTER_KAPPA}" \
  --w-riccati-center-peak "${W_RICCATI_CENTER_PEAK}" \
  --w-riccati-boundary-band-kappa "${W_RICCATI_BOUNDARY_BAND_KAPPA}" \
  --w-riccati-boundary-band-q "${W_RICCATI_BOUNDARY_BAND_Q}" \
  --riccati-boundary-band-points 32 \
  --riccati-boundary-band-start 0.94 \
  --riccati-boundary-band-end 0.995 \
  --w-ci-stability-outside 0.0 \
  --w-ci-neutrality 0.0 \
  --w-ci-low-alpha-zero "${W_CI_LOW_ALPHA_ZERO}" \
  --w-ci-smoothness "${W_CI_SMOOTHNESS}" \
  --n-ci-spectral-grid "${N_CI_SPECTRAL_GRID}" \
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
  --device "${DEVICE:-cuda}" \
  --output-dir "${OUTPUT_DIR:-model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference}"
