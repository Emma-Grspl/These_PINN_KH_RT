#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="${ROOT_DIR}/assets/blumen_gep"
PLOT_DIR="${ROOT_DIR}/plot_presentation/supersonic_classique"

OUTPUT_STEM="${1:-supersonic_global_m10_18_a00_03}"

CHUNK_STEMS=(
  "${OUTPUT_STEM}_chunk00"
  "${OUTPUT_STEM}_chunk01"
  "${OUTPUT_STEM}_chunk02"
  "${OUTPUT_STEM}_chunk03"
)

echo "Fusion des chunks vers ${OUTPUT_STEM}"
python "${ROOT_DIR}/classical_solver/gep/merge_supersonic_mode_database_chunks.py" \
  --chunk-stems "${CHUNK_STEMS[@]}" \
  --output-stem "${OUTPUT_STEM}"

SURFACE_CSV="${ASSETS_DIR}/${OUTPUT_STEM}_surface.csv"

echo "Analyse diagnostique a partir de ${SURFACE_CSV}"
python "${ROOT_DIR}/scripts/analyze_supersonic_mode_database.py" \
  --surface-csv "${SURFACE_CSV}" \
  --output-dir "${PLOT_DIR}"

echo "Plots presentation a partir de ${SURFACE_CSV}"
python "${ROOT_DIR}/scripts/plot_supersonic_database_presentation.py" \
  --surface-csv "${SURFACE_CSV}" \
  --output-dir "${PLOT_DIR}"

echo
echo "Surface CSV : ${SURFACE_CSV}"
echo "Dossier figures : ${PLOT_DIR}"
