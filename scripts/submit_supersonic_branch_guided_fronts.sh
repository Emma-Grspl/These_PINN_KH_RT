#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# This launcher only works when the target Mach is bracketed by two existing
# reference Mach lines in the central supersonic reference package.
MACH_VALUES=(${MACH_VALUES:-1.65 1.70 1.75})
ALPHA_VALUES=(${ALPHA_VALUES:-0.100000 0.125000 0.150000})
REFERENCE_CSV="${REFERENCE_CSV:-assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv}"

WORKERS="${WORKERS:-16}"
ALPHA_TOLERANCE="${ALPHA_TOLERANCE:-5e-4}"
MATCH_Y="${MATCH_Y:-1.0}"
MAPPING_SCALE="${MAPPING_SCALE:-5.0}"
MIN_Y_LIMIT="${MIN_Y_LIMIT:-10.0}"
MAX_Y_LIMIT="${MAX_Y_LIMIT:-800.0}"
Y_LIMIT_FACTOR="${Y_LIMIT_FACTOR:-8.0}"
AMP_LOWER_BOUND="${AMP_LOWER_BOUND:--30.0}"
AMP_UPPER_BOUND="${AMP_UPPER_BOUND:-5.0}"
CR_HALF_WINDOWS="${CR_HALF_WINDOWS:-0.010 0.020 0.040 0.080}"
CI_HALF_WINDOWS="${CI_HALF_WINDOWS:-0.004 0.008 0.015 0.030}"
RETRY_GROWTH="${RETRY_GROWTH:-1.50}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_ITER="${MAX_ITER:-12}"
GRID_SIZE="${GRID_SIZE:-5}"
CI_WEIGHT="${CI_WEIGHT:-6.0}"
CR_WEIGHT="${CR_WEIGHT:-0.25}"
CONTINUITY_WEIGHT="${CONTINUITY_WEIGHT:-0.50}"
INCLUDE_GENERIC_SEEDS="${INCLUDE_GENERIC_SEEDS:-0}"

echo "Soumission campagne supersonique branch-guided fronts"
echo "Mach values: ${MACH_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "workers=${WORKERS}"

if [[ ! -f "${REFERENCE_CSV}" ]]; then
  echo "Reference CSV introuvable: ${REFERENCE_CSV}" >&2
  exit 1
fi

min_ref_mach="$(python3 - "${REFERENCE_CSV}" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
vals = sorted(float(v) for v in df["Mach"].unique())
print(vals[0])
PY
)"
max_ref_mach="$(python3 - "${REFERENCE_CSV}" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
vals = sorted(float(v) for v in df["Mach"].unique())
print(vals[-1])
PY
)"

echo "Reference Mach bracket available: (${min_ref_mach}, ${max_ref_mach})"

for mach in "${MACH_VALUES[@]}"; do
  if ! python3 - "${mach}" "${min_ref_mach}" "${max_ref_mach}" <<'PY'
import sys
target = float(sys.argv[1])
min_ref = float(sys.argv[2])
max_ref = float(sys.argv[3])
sys.exit(0 if (min_ref < target < max_ref) else 1)
PY
  then
    echo "Skipping M=${mach}: branch-guided requires two bracketing reference Mach lines, but current reference only covers (${min_ref_mach}, ${max_ref_mach})." >&2
    echo "Use the non-guided pointwise launcher instead: POINTS=\"0.100000:${mach} 0.125000:${mach} 0.150000:${mach}\" sbatch launch/jz_submit_supersonic_shooting_point_batch.slurm" >&2
    continue
  fi

  alphas_csv="$(IFS=,; echo "${ALPHA_VALUES[*]}")"
  mach_tag="$(printf '%0.2f' "${mach}" | tr -d '.')"
  output_stem="supersonic_shooting_branch_guided_front_M${mach_tag}"

  echo "Submitting M=${mach} alpha=${alphas_csv} -> ${output_stem}"
  WORKERS="${WORKERS}" \
  MACH="${mach}" \
  ALPHAS="${alphas_csv}" \
  ALPHA_TOLERANCE="${ALPHA_TOLERANCE}" \
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
  INCLUDE_GENERIC_SEEDS="${INCLUDE_GENERIC_SEEDS}" \
  OUTPUT_STEM="${output_stem}" \
  sbatch launch/jz_submit_supersonic_shooting_point_batch_M140_branch_guided.slurm
done
