#!/bin/bash
set -euo pipefail

export W_RICCATI_SHOOTING_PATH="${W_RICCATI_SHOOTING_PATH:-10.0}"
export RICCATI_SHOOTING_PATH_POINTS="${RICCATI_SHOOTING_PATH_POINTS:-33}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash scripts/run_kh_subsonic_singlecase_pure_physics_scalar_ci_matching_phaseA_bis.sh
