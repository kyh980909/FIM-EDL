#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/yongho/miniconda3/envs/fedl/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

SEEDS="${SEEDS:-0}"
SUITE="${SUITE:-cifar_additional_20260319}"
RUN_FIXED="${RUN_FIXED:-1}"
RUN_CONTROLLER="${RUN_CONTROLLER:-1}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"

ARGS=(--suite "${SUITE}" --seeds ${SEEDS})
if [[ "${RUN_FIXED}" == "1" ]]; then
  ARGS+=(--run-fixed)
fi
if [[ "${RUN_CONTROLLER}" == "1" ]]; then
  ARGS+=(--run-controller)
fi
if [[ "${RUN_SENSITIVITY}" == "1" ]]; then
  ARGS+=(--run-sensitivity)
fi

"${PYTHON_BIN}" scripts/paper/run_additional_cifar_experiments.py "${ARGS[@]}"
"${PYTHON_BIN}" scripts/paper/export_additional_experiments.py --runs runs --out results/additional_experiments
