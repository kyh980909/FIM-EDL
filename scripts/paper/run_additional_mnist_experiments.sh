#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/yongho/miniconda3/envs/fedl/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

SUITE="${SUITE:-mnist_additional_20260320}"
SEEDS="${SEEDS:-0}"
RUN_CONTROLLER="${RUN_CONTROLLER:-1}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"
RUN_FIXED_BASELINE="${RUN_FIXED_BASELINE:-0}"
ARGS=(--suite "${SUITE}" --seeds ${SEEDS})
if [[ "${RUN_FIXED_BASELINE}" == "1" ]]; then
  ARGS+=(--run-fixed)
fi
if [[ "${RUN_CONTROLLER}" == "1" ]]; then
  ARGS+=(--run-controller)
fi
if [[ "${RUN_SENSITIVITY}" == "1" ]]; then
  ARGS+=(--run-sensitivity)
fi

"${PYTHON_BIN}" scripts/paper/run_additional_mnist_experiments.py "${ARGS[@]}"

echo "[EXPORT] additional experiment summaries"
"${PYTHON_BIN}" scripts/paper/export_additional_experiments.py --runs runs --out results/additional_experiments
