#!/usr/bin/env bash
set -euo pipefail

cd "$WORK/These_PINN_KH_RT"

MISSING_TSV="${1:?Usage: ./submit_missing_sparse_supersonic_expand1.sh missing.tsv}"
QUEUE_CAP="${QUEUE_CAP:-50}"

SUBMIT_LOG="manual_single_launch_sparse_supersonic_expand1_missing_submit_$(date +%Y%m%d_%H%M%S).tsv"
echo -e "job_id\talpha\tMach\tseed\tstem\tcampaign" > "$SUBMIT_LOG"

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

tail -n +2 "$MISSING_TSV" | while IFS=$'\t' read -r A M SEED STEM CAMPAIGN_FIELD; do
  [ -n "$STEM" ] || continue

  case "$STEM" in
    *M180_lowalpha*) CR_HW="0.035"; CI_HW="0.010" ;;
    *M180_core*)     CR_HW="0.030"; CI_HW="0.014" ;;
    *M12to16_core*)  CR_HW="0.035"; CI_HW="0.014" ;;
    *M110_catch*)    CR_HW="0.040"; CI_HW="0.006" ;;
    *M190_catch*)    CR_HW="0.035"; CI_HW="0.018" ;;
    *)               CR_HW="0.035"; CI_HW="0.014" ;;
  esac

  wait_for_queue_slot

  jid=$(POINTS="${A}:${M}" \
    OUTPUT_STEM="${STEM}" \
    CANDIDATE_SOURCE=manual_grid \
    MANUAL_SEED_GRID="${SEED}" \
    MAX_CANDIDATES_PER_POINT=1 \
    MAX_RETRIES=1 \
    MAX_ITER=100 \
    GRID_SIZE=17 \
    CR_HALF_WINDOWS="${CR_HW}" \
    CI_HALF_WINDOWS="${CI_HW}" \
    EXISTING_MACH_WINDOW=0.35 \
    BOX_REQUIRED=0 \
    REQUIRE_BOX=0 \
    DRY_RUN_CANDIDATES=0 \
    sbatch --parsable launch/jz_submit_supersonic_shooting_blumen_locked_multicandidate.slurm)

  echo -e "${jid}\t${A}\t${M}\t${SEED}\t${STEM}\t${CAMPAIGN_FIELD}" | tee -a "$SUBMIT_LOG"
  sleep 0.2
done

echo "[done]"
echo "$SUBMIT_LOG"
