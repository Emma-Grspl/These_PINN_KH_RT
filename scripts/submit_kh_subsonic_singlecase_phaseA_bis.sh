#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

launcher="launch/jz_submit_kh_subsonic_pinn_singlecase_riccati_pure_scalar_ci_matching_phaseA_bis_param.slurm"

declare -a phase_a_cases=(
  "0.25 0.50"
  "0.65 0.50"
  "0.25 0.60"
  "0.65 0.60"
)

for spec in "${phase_a_cases[@]}"; do
  read -r alpha mach <<<"${spec}"
  alpha_tag="$(printf '%0.3f' "${alpha}" | tr -d '.')"
  mach_tag="$(printf '%0.3f' "${mach}" | tr -d '.')"
  output_dir="model_saved/kh_subsonic_singlecase_phaseA_bis_a${alpha_tag}_m${mach_tag}_scalar_ci_matching"

  echo "Submitting Phase A-bis case alpha=${alpha} mach=${mach} -> ${output_dir}"
  ALPHA_VALUE="${alpha}" MACH_VALUE="${mach}" OUTPUT_DIR="${output_dir}" sbatch "${launcher}"
done
