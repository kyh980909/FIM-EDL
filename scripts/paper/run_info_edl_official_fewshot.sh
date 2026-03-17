#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yongho/FIM-EDL"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/tmp/IEDL_official/code_fsl}"
PYTHON_BIN="${PYTHON_BIN:-/home/yongho/miniconda3/envs/fedl/bin/python}"

METHOD="${METHOD:-infoedl}"
WAYS="${WAYS:-5}"
TASKS="${TASKS:-10000}"
TASK_START="${TASK_START:-0}"
SPLIT="${SPLIT:-novel}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/fewshot_official/raw}"
LOG_PATH="${LOG_PATH:-$ROOT/outputs/logs/${METHOD}_${WAYS}w_official.log}"
CONFIG_ID="${CONFIG_ID:-${WAYS}w-${METHOD}-official}"
SUFFIX="${SUFFIX:-official}"
DUMP_PERIOD="${DUMP_PERIOD:-10000}"
USE_WANDB="${USE_WANDB:-false}"
TORCH_THREADS="${TORCH_THREADS:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adam}"
INFO_BETA="${INFO_BETA:-1.0}"
INFO_GAMMA="${INFO_GAMMA:-1.0}"
LBFGS_LR="${LBFGS_LR:-0.25}"
LBFGS_LINE_SEARCH_FN="${LBFGS_LINE_SEARCH_FN:-none}"
ADAM_LR="${ADAM_LR:-0.01}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

mkdir -p "$(dirname "$LOG_PATH")" "$RESULTS_DIR"

cd "$OFFICIAL_ROOT"

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS
export NUMEXPR_NUM_THREADS

"$PYTHON_BIN" "$ROOT/scripts/paper/run_official_fewshot.py" \
  --official-root "$OFFICIAL_ROOT" \
  --method "$METHOD" \
  --ways "$WAYS" \
  --tasks "$TASKS" \
  --task-start "$TASK_START" \
  --split "$SPLIT" \
  --results-dir "$RESULTS_DIR" \
  --config-id "$CONFIG_ID" \
  --suffix "$SUFFIX" \
  --dump-period "$DUMP_PERIOD" \
  --torch-threads "$TORCH_THREADS" \
  --optimizer-name "$OPTIMIZER_NAME" \
  --info-beta "$INFO_BETA" \
  --info-gamma "$INFO_GAMMA" \
  --lbfgs-lr "$LBFGS_LR" \
  --lbfgs-line-search-fn "$LBFGS_LINE_SEARCH_FN" \
  --adam-lr "$ADAM_LR" \
  --grad-clip-norm "$GRAD_CLIP_NORM" \
  $( [ "$USE_WANDB" = "true" ] && printf '%s' --use-wandb ) \
  >>"$LOG_PATH" 2>&1
