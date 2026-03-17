#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/yongho/FIM-EDL"
cd "$ROOT_DIR"

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

PRESET="${PRESET:-info_edl_beta005_gamma20_fewshot_train}"
SHOTS="${SHOTS:-1 5 20}"
WAYS="${WAYS:-5 10}"
EPISODES="${EPISODES:-200}"
ADAPT_STEPS="${ADAPT_STEPS:-10}"
ADAPT_LR="${ADAPT_LR:-0.01}"
LOG_PATH="${LOG_PATH:-outputs/logs/info_edl_beta005_gamma20_fewshot.log}"

mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "[START] $(date -Iseconds)"
  echo "[INFO] preset=$PRESET shots=$SHOTS ways=$WAYS episodes=$EPISODES adapt_steps=$ADAPT_STEPS"

  "$PYTHON_BIN" run.py preset "$PRESET"

  CKPT_PATH="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

run_root = Path('runs') / 'info_edl' / 'seed_0'
latest = sorted(run_root.glob('*/summary.json'))[-1]
payload = json.loads(latest.read_text(encoding='utf-8'))
print(payload['summary']['best_model_path'])
PY
)"

  echo "[INFO] checkpoint=$CKPT_PATH"

  for way in $WAYS; do
    for shot in $SHOTS; do
      echo "[EVAL] way=$way shot=$shot"
      CKPT_PATH="$CKPT_PATH" \
      WAY="$way" \
      SHOT="$shot" \
      EPISODES="$EPISODES" \
      ADAPT_STEPS="$ADAPT_STEPS" \
      ADAPT_LR="$ADAPT_LR" \
      LOG_PATH="outputs/logs/info_edl_b005_g20_${way}way_${shot}shot_${EPISODES}ep.log" \
      PYTHON_BIN="$PYTHON_BIN" bash scripts/run_info_edl_fewshot_eval.sh
    done
  done

  "$PYTHON_BIN" scripts/paper/export_fewshot_results.py --runs runs --out results/fewshot
  "$PYTHON_BIN" scripts/paper/build_iedl_table4.py --runs runs --methods info_edl --out-dir results/paper_tables/iedl_table4
  echo "[DONE] $(date -Iseconds)"
} >> "$LOG_PATH" 2>&1
