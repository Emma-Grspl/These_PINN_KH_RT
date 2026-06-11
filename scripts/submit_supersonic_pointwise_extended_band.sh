#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MACH_VALUES=(${MACH_VALUES:-1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8})
ALPHA_VALUES=(${ALPHA_VALUES:-0.050000 0.075000 0.100000 0.125000 0.150000 0.175000 0.200000 0.225000 0.250000})

WORKERS="${WORKERS:-16}"
MATCH_Y="${MATCH_Y:-1.0}"
MAPPING_SCALE="${MAPPING_SCALE:-5.0}"
MIN_Y_LIMIT="${MIN_Y_LIMIT:-10.0}"
MAX_Y_LIMIT="${MAX_Y_LIMIT:-800.0}"
Y_LIMIT_FACTOR="${Y_LIMIT_FACTOR:-8.0}"
AMP_LOWER_BOUND="${AMP_LOWER_BOUND:--30.0}"
AMP_UPPER_BOUND="${AMP_UPPER_BOUND:-5.0}"
CR_HALF_WINDOWS="${CR_HALF_WINDOWS:-0.015 0.03 0.06 0.10}"
CI_HALF_WINDOWS="${CI_HALF_WINDOWS:-0.008 0.015 0.03}"
RETRY_GROWTH="${RETRY_GROWTH:-1.75}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_ITER="${MAX_ITER:-10}"
GRID_SIZE="${GRID_SIZE:-4}"
CI_WEIGHT="${CI_WEIGHT:-4.0}"
CR_WEIGHT="${CR_WEIGHT:-0.35}"
CONTINUITY_WEIGHT="${CONTINUITY_WEIGHT:-0.20}"

echo "Soumission campagne etendue supersonique"
echo "Mach values: ${MACH_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"

for mach in "${MACH_VALUES[@]}"; do
  points=()
  for alpha in "${ALPHA_VALUES[@]}"; do
    points+=("${alpha}:${mach}")
  done
  points_string="${points[*]}"
  output_stem="$(printf 'supersonic_shooting_extended_band_M%s' "${mach}" | tr -d '.')"

  echo "Submitting M=${mach} -> ${output_stem}"
  WORKERS="${WORKERS}" \
  POINTS="${points_string}" \
  MATCH_Y="${MATCH_Y}" \
  MAPPING_SCALE="${MAPPING_SCALE}" \
  MIN_Y_LIMIT="${MIN_Y_LIMIT}" \
  MAX_Y_LIMIT="${MAX_Y_LIMIT}" \
  Y_LIMIT_FACTOR="${Y_LIMIT_FACTOR}" \
  AMP_LOWER_BOUND="${AMP_LOWER_BOUND}" \
  AMP_UPPER_BOUND="${AMP_UPPER_BOUND}" \
  CR_HALF_WINDOWS="${CR_HALF_WINDOWS}" \
  CI_HALF_WINDOWS="${CI_HALF_WINDOWS}" \
  RETRY_GROWTH="${RETRY_GROWTH}" \
  MAX_RETRIES="${MAX_RETRIES}" \
  MAX_ITER="${MAX_ITER}" \
  GRID_SIZE="${GRID_SIZE}" \
  CI_WEIGHT="${CI_WEIGHT}" \
  CR_WEIGHT="${CR_WEIGHT}" \
  CONTINUITY_WEIGHT="${CONTINUITY_WEIGHT}" \
  OUTPUT_STEM="${output_stem}" \
  sbatch launch/jz_submit_supersonic_shooting_point_batch.slurm
done
