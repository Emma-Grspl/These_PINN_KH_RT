#!/usr/bin/env bash
set -euo pipefail

cd "$WORK/These_PINN_KH_RT"
mkdir -p slurm/log

CAMPAIGN="${CAMPAIGN:-sparse_supersonic_expand1}"
QUEUE_CAP="${QUEUE_CAP:-50}"
DRY_SUBMIT="${DRY_SUBMIT:-0}"

MAX_RETRIES="${MAX_RETRIES:-1}"
MAX_ITER="${MAX_ITER:-100}"
GRID_SIZE="${GRID_SIZE:-17}"

LAUNCH_LOG="manual_single_launch_${CAMPAIGN}_$(date +%Y%m%d_%H%M%S).tsv"
echo -e "job_id\talpha\tMach\tseed\tstem\tcampaign" > "$LAUNCH_LOG"

tag_float() {
  printf "%s" "$1" | sed 's/-/m/g; s/\.//g'
}

wait_for_queue_slot() {
  while true; do
    n=$(squeue -u "$USER" -h -n KH_shoot_multi 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -lt "$QUEUE_CAP" ]; then
      break
    fi
    echo "[queue] $n KH_shoot_multi jobs >= QUEUE_CAP=$QUEUE_CAP ; sleeping 60s"
    sleep 60
  done
}

submit_one() {
  local A="$1"
  local M="$2"
  local SEED="$3"
  local FAMILY="$4"
  local CR_HW="$5"
  local CI_HW="$6"

  local CR="${SEED%:*}"
  local CI="${SEED#*:}"

  local A_TAG
  local M_TAG
  local CR_TAG
  local CI_TAG

  A_TAG="$(tag_float "$A")"
  M_TAG="$(tag_float "$M")"
  CR_TAG="$(tag_float "$CR")"
  CI_TAG="$(tag_float "$CI")"

  local STEM="supersonic_multicandidate_M${M_TAG}_a${A_TAG}_${CAMPAIGN}_${FAMILY}_cr${CR_TAG}_ci${CI_TAG}"

  if [ "$DRY_SUBMIT" = "1" ]; then
    echo -e "DRY\t$A\t$M\t$SEED\t$STEM\t$CAMPAIGN" | tee -a "$LAUNCH_LOG"
    return 0
  fi

  wait_for_queue_slot

  jid=$(POINTS="${A}:${M}" \
    OUTPUT_STEM="${STEM}" \
    CANDIDATE_SOURCE=manual_grid \
    MANUAL_SEED_GRID="${SEED}" \
    MAX_CANDIDATES_PER_POINT=1 \
    MAX_RETRIES="${MAX_RETRIES}" \
    MAX_ITER="${MAX_ITER}" \
    GRID_SIZE="${GRID_SIZE}" \
    CR_HALF_WINDOWS="${CR_HW}" \
    CI_HALF_WINDOWS="${CI_HW}" \
    EXISTING_MACH_WINDOW=0.35 \
    BOX_REQUIRED=0 \
    REQUIRE_BOX=0 \
    DRY_RUN_CANDIDATES=0 \
    sbatch --parsable launch/jz_submit_supersonic_shooting_blumen_locked_multicandidate.slurm)

  echo -e "${jid}\t${A}\t${M}\t${SEED}\t${STEM}\t${CAMPAIGN}" | tee -a "$LAUNCH_LOG"
  sleep 0.2
}

echo "[campaign] $CAMPAIGN"
echo "[queue cap] $QUEUE_CAP"
echo "[launch log] $LAUNCH_LOG"

# --------------------------------------------------------------------
# M = 1.8 : nouveaux alpha demandés.
# alpha=0.100, 0.150 : graines avec ci plus faible.
# alpha=0.225, 0.300 : graines autour de la zone qui a déjà marché.
# --------------------------------------------------------------------

for A in 0.100 0.150; do
  for SEED in \
    "0.24:0.0015" \
    "0.30:0.0040" \
    "0.36:0.0100" \
    "0.42:0.0160"
  do
    submit_one "$A" "1.800" "$SEED" "M180_lowalpha" "0.035" "0.010"
  done
done

for A in 0.225 0.300; do
  for SEED in \
    "0.36:0.0120" \
    "0.40:0.0240" \
    "0.42:0.0120" \
    "0.44:0.0240"
  do
    submit_one "$A" "1.800" "$SEED" "M180_core" "0.030" "0.014"
  done
done

# --------------------------------------------------------------------
# M = 1.2, 1.3, 1.4, 1.5, 1.6
# alpha entre 0.2 et 0.3.
# Premier passage coarse : 5 alpha x 5 Mach x 4 seeds = 100 jobs.
# --------------------------------------------------------------------

for M in 1.200 1.300 1.400 1.500 1.600; do
  for A in 0.200 0.225 0.250 0.275 0.300; do
    for SEED in \
      "0.32:0.0040" \
      "0.36:0.0120" \
      "0.40:0.0200" \
      "0.44:0.0120"
    do
      submit_one "$A" "$M" "$SEED" "M12to16_core" "0.035" "0.014"
    done
  done
done

# --------------------------------------------------------------------
# Tentatives M = 1.1 : plus difficile/proche seuil, ci potentiellement faible.
# On commence avec alpha=0.20, 0.25, 0.30.
# --------------------------------------------------------------------

for A in 0.200 0.250 0.300; do
  for SEED in \
    "0.20:0.0010" \
    "0.28:0.0025" \
    "0.36:0.0060" \
    "0.44:0.0100"
  do
    submit_one "$A" "1.100" "$SEED" "M110_catch" "0.040" "0.006"
  done
done

# --------------------------------------------------------------------
# Tentatives M = 1.9 : extension haute Mach.
# On commence avec alpha=0.20, 0.25, 0.30.
# --------------------------------------------------------------------

for A in 0.200 0.250 0.300; do
  for SEED in \
    "0.36:0.0160" \
    "0.40:0.0260" \
    "0.44:0.0160" \
    "0.44:0.0340"
  do
    submit_one "$A" "1.900" "$SEED" "M190_catch" "0.035" "0.018"
  done
done

echo
echo "[done launching]"
echo "$LAUNCH_LOG"
