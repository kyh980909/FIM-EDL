#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yongho/FIM-EDL"

METHOD="${METHOD:-infoedl}"
WAYS="${WAYS:-5}"
INFO_BETA="${INFO_BETA:-1.0}"
INFO_GAMMA="${INFO_GAMMA:-1.0}"
TOTAL_TASKS="${TOTAL_TASKS:-10000}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
BASE_CONFIG_ID="${BASE_CONFIG_ID:-${WAYS}w-${METHOD}-official-chunked}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/fewshot_official/raw}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

task_start=0
while [ "$task_start" -lt "$TOTAL_TASKS" ]; do
  remaining=$((TOTAL_TASKS - task_start))
  if [ "$remaining" -lt "$CHUNK_SIZE" ]; then
    chunk_tasks="$remaining"
  else
    chunk_tasks="$CHUNK_SIZE"
  fi

  log_path="$LOG_DIR/${BASE_CONFIG_ID}_start${task_start}.log"

  env \
    METHOD="$METHOD" \
    WAYS="$WAYS" \
    INFO_BETA="$INFO_BETA" \
    INFO_GAMMA="$INFO_GAMMA" \
    TASKS="$chunk_tasks" \
    TASK_START="$task_start" \
    DUMP_PERIOD="$chunk_tasks" \
    RESULTS_DIR="$RESULTS_DIR" \
    LOG_PATH="$log_path" \
    CONFIG_ID="$BASE_CONFIG_ID" \
    SUFFIX=official \
    bash "$ROOT/scripts/paper/run_info_edl_official_fewshot.sh"

  task_start=$((task_start + chunk_tasks))
done
