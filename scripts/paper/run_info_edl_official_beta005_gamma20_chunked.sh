#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yongho/FIM-EDL"

WAYS="${WAYS:-5}"
TOTAL_TASKS="${TOTAL_TASKS:-10000}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/fewshot_official_beta005_gamma20/raw}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs}"
BASE_CONFIG_ID="${BASE_CONFIG_ID:-${WAYS}w-infoedl-beta005-gamma20-official}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

env \
  METHOD=infoedl \
  WAYS="$WAYS" \
  INFO_BETA=0.05 \
  INFO_GAMMA=2.0 \
  TOTAL_TASKS="$TOTAL_TASKS" \
  CHUNK_SIZE="$CHUNK_SIZE" \
  BASE_CONFIG_ID="$BASE_CONFIG_ID" \
  RESULTS_DIR="$RESULTS_DIR" \
  LOG_DIR="$LOG_DIR" \
  OPTIMIZER_NAME="${OPTIMIZER_NAME:-adam}" \
  ADAM_LR="${ADAM_LR:-0.01}" \
  GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}" \
  bash "$ROOT/scripts/paper/run_info_edl_official_fewshot_chunked.sh"
