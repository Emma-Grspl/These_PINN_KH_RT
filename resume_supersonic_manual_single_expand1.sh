#!/usr/bin/env bash
set -euo pipefail

cd "$WORK/These_PINN_KH_RT"
mkdir -p slurm/log

CAMPAIGN="${CAMPAIGN:-sparse_supersonic_expand1}"
QUEUE_CAP="${QUEUE_CAP:-50}"

PLAN_LOG="$(grep -l '^DRY' manual_single_launch_${CAMPAIGN}_*.tsv | head -1)"

if [ -z "${PLAN_LOG:-}" ] || [ ! -f "$PLAN_LOG" ]; then
  echo "Could not find DRY plan log."
  exit 1
fi

RESUME_LOG="manual_single_launch_${CAMPAIGN}_resume_$(date +%Y%m%d_%H%M%S).tsv"
echo -e "job_id\talpha\tMach\tseed\tstem\tcampaign" > "$RESUME_LOG"

echo "[plan]   $PLAN_LOG"
echo "[resume] $RESUME_LOG"

already_done_stems_file="$(mktemp)"
trap 'rm -f "$already_done_stems_file"' EXIT

for f in manual_single_launch_${CAMPAIGN}_*.tsv; do
  [ -f "$f" ] || continue
  awk -F '\t' 'NR>1 && $1 ~ /^[0-9]+$/ {print $5}' "$f" >> "$already_done_stems_file" || true
done

find assets/classic_supersonic/multicandidate_audits \
  -maxdepth 1 -type d -name "supersonic_multicandidate_*_${CAMPAIGN}_*" \
  -printf "%f\n" >> "$already_done_stems_file" 2>/dev/null || true

sort -u "$already_done_stems_file" -o "$already_done_stems_file"

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

tail -n +2 "$PLAN_LOG" | while IFS=$'\t' read -r _ A M SEED STEM CAMPAIGN_FIELD; do
  [ -n "$STEM" ] || continue

  if grep -Fxq "$STEM" "$already_done_stems_file"; then
    echo "[skip] $STEM"
    continue
  fi

  wait_for_queue_slot

  jid=$(POINTS="${A}:${M}" \
    OUTPUT_STEM="${STEM}" \
    CANDIDATE_SOURCE=manual_grid \
    MANUAL_SEED_GRID="${SEED}" \
    MAX_CANDIDATES_PER_POINT=1 \
    MAX_RETRIES=1 \
    MAX_ITER=100 \
    GRID_SIZE=17 \
    CR_HALF_WINDOWS="0.035" \
    CI_HALF_WINDOWS="0.014" \
    EXISTING_MACH_WINDOW=0.35 \
    BOX_REQUIRED=0 \
    REQUIRE_BOX=0 \
    DRY_RUN_CANDIDATES=0 \
    sbatch --parsable launch/jz_submit_supersonic_shooting_blumen_locked_multicandidate.slurm)

  echo -e "${jid}\t${A}\t${M}\t${SEED}\t${STEM}\t${CAMPAIGN}" | tee -a "$RESUME_LOG"
  sleep 0.2
done

echo "[done]"
echo "resume log: $RESUME_LOG"
