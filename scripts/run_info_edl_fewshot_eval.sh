#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CKPT_PATH="${CKPT_PATH:-/home/yongho/FIM-EDL/runs/info_edl/seed_0/20260317T062939546673Z/checkpoints/best.ckpt}"
WAY="${WAY:-5}"
SHOT="${SHOT:-1}"
EPISODES="${EPISODES:-200}"
ADAPT_STEPS="${ADAPT_STEPS:-10}"
ADAPT_LR="${ADAPT_LR:-0.01}"
BACKBONE="${BACKBONE:-resnet18}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:-}"
NUM_CLASSES="${NUM_CLASSES:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
OOD_DATASET="${OOD_DATASET:-cub}"
STRICT="${STRICT:-false}"
LOG_PATH="${LOG_PATH:-outputs/logs/info_edl_fewshot_${WAY}way_${SHOT}shot_${EPISODES}ep.log}"

mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "[START] $(date -Iseconds)"
  echo "[INFO] checkpoint=$CKPT_PATH way=$WAY shot=$SHOT episodes=$EPISODES adapt_steps=$ADAPT_STEPS backbone=$BACKBONE"

  uv run python -m src.eval_fewshot \
    experiment=info_edl \
    seed=0 \
    checkpoint="$CKPT_PATH" \
    experiment.suite=fewshot_paper \
    model.backbone="$BACKBONE" \
    model.backbone_checkpoint="$BACKBONE_CHECKPOINT" \
    model.num_classes="$NUM_CLASSES" \
    data.id=miniimagenet \
    "data.ood_list=[$OOD_DATASET]" \
    data.batch_size="$BATCH_SIZE" \
    data.num_workers="$NUM_WORKERS" \
    trainer.accelerator=cpu \
    trainer.precision=32 \
    logging.backend=csv \
    fewshot.enabled=true \
    fewshot.paper_protocol_strict="$STRICT" \
    fewshot.eval_split="$EVAL_SPLIT" \
    fewshot.way="$WAY" \
    fewshot.shot="$SHOT" \
    fewshot.episodes="$EPISODES" \
    fewshot.adapt_steps="$ADAPT_STEPS" \
    fewshot.adapt_lr="$ADAPT_LR"

  uv run python scripts/paper/export_fewshot_results.py --runs runs --out results/fewshot
  echo "[DONE] $(date -Iseconds)"
} >> "$LOG_PATH" 2>&1
