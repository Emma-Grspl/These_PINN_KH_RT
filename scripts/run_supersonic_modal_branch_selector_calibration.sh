#!/usr/bin/env bash
set -euo pipefail

run_case() {
  local suffix="$1"
  local w_shooting="$2"
  local w_reference_overlap="$3"
  local w_previous_overlap="$4"
  local w_centroid="$5"
  local w_spread="$6"
  local w_phase_span="$7"
  local w_cr_jump="$8"
  local w_ci_jump="$9"
  local soft_cr_floor="${10}"
  local w_cr_floor="${11}"

  echo
  echo "===== Calibration case: ${suffix} ====="
  echo "w_shooting=${w_shooting} w_reference_overlap=${w_reference_overlap} w_previous_overlap=${w_previous_overlap}"
  echo "w_centroid=${w_centroid} w_spread=${w_spread} w_phase_span=${w_phase_span}"
  echo "w_cr_jump=${w_cr_jump} w_ci_jump=${w_ci_jump} soft_cr_floor=${soft_cr_floor} w_cr_floor=${w_cr_floor}"

  python3 scripts/diagnose_supersonic_modal_branch_selector.py \
    --alpha "${ALPHA_VALUE:-0.20}" \
    --mach-values ${MACH_VALUES:-1.20 1.25 1.275 1.30} \
    --reference-mach "${REFERENCE_MACH:-1.30}" \
    --n-points "${N_POINTS:-561}" \
    --mapping-kind "${MAPPING_KIND:-pin}" \
    --mapping-scale "${MAPPING_SCALE:-1.5}" \
    --cubic-delta "${CUBIC_DELTA:-0.2}" \
    --xi-max "${XI_MAX:-0.90}" \
    --ci-weight "${CI_WEIGHT:-2.0}" \
    --high-cr-threshold "${HIGH_CR_THRESHOLD:-0.60}" \
    --candidate-top-k "${CANDIDATE_TOP_K:-80}" \
    --distance-tol "${DISTANCE_TOL:-0.02}" \
    --w-shooting "${w_shooting}" \
    --w-reference-overlap "${w_reference_overlap}" \
    --w-previous-overlap "${w_previous_overlap}" \
    --w-centroid "${w_centroid}" \
    --w-spread "${w_spread}" \
    --w-phase-span "${w_phase_span}" \
    --w-cr-jump "${w_cr_jump}" \
    --w-ci-jump "${w_ci_jump}" \
    --soft-cr-floor "${soft_cr_floor}" \
    --w-cr-floor "${w_cr_floor}" \
    --output-stem "${OUTPUT_STEM_PREFIX:-supersonic_modal_branch_selector_calib_a020_m120_130}_${suffix}"
}

run_case "refheavy" "0.40" "1.80" "0.80" "0.50" "0.30" "0.30" "0.15" "0.05" "0.00" "0.00"
run_case "refheavy_floor058" "0.35" "1.90" "0.90" "0.50" "0.30" "0.30" "0.15" "0.05" "0.58" "1.00"
run_case "refdom_floor060" "0.25" "2.20" "1.00" "0.45" "0.25" "0.25" "0.10" "0.05" "0.60" "1.40"
