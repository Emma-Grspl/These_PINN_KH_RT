#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

ALPHA="${ALPHA:-0.26}"
MACH="${MACH:-1.8}"
CR="${CR:-0.38}"
CI="${CI:-0.024}"
OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_supersonic_singlecase_pressure_fixed_c}"
EPOCHS="${EPOCHS:-3000}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
N_INTERIOR="${N_INTERIOR:-512}"
N_BOUNDARY="${N_BOUNDARY:-64}"
N_CENTER="${N_CENTER:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
DEPTH="${DEPTH:-6}"
ACTIVATION="${ACTIVATION:-tanh}"
W_PDE="${W_PDE:-1.0}"
W_BC="${W_BC:-20.0}"
W_GAUGE="${W_GAUGE:-100.0}"
W_CENTER_PDE="${W_CENTER_PDE:-1.0}"
YMAX="${YMAX:-120.0}"
ENVELOPE_EPS="${ENVELOPE_EPS:-1.0}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-1234}"
AUDIT_EVERY="${AUDIT_EVERY:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

echo "Running supersonic single-case pressure-first PINN"
echo "alpha=${ALPHA} mach=${MACH} cr=${CR} ci=${CI}"
echo "output_dir=${OUTPUT_DIR}"
echo "epochs=${EPOCHS} lr=${LEARNING_RATE}"
echo "weights: pde=${W_PDE} bc=${W_BC} gauge=${W_GAUGE} center=${W_CENTER_PDE}"
echo "ymax=${YMAX} envelope_eps=${ENVELOPE_EPS}"

"${PYTHON_BIN}" scripts/train_kh_supersonic_singlecase_pressure_fixed_c.py \
  --alpha "${ALPHA}" \
  --mach "${MACH}" \
  --cr "${CR}" \
  --ci "${CI}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --n-interior "${N_INTERIOR}" \
  --n-boundary "${N_BOUNDARY}" \
  --n-center "${N_CENTER}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --depth "${DEPTH}" \
  --activation "${ACTIVATION}" \
  --w-pde "${W_PDE}" \
  --w-bc "${W_BC}" \
  --w-gauge "${W_GAUGE}" \
  --w-center-pde "${W_CENTER_PDE}" \
  --ymax "${YMAX}" \
  --envelope-eps "${ENVELOPE_EPS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --audit-every "${AUDIT_EVERY}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}"
