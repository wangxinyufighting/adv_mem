#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_PATH=${ANSWER_AGENT_MODEL:-Qwen/Qwen3-0.6B}
PORT=${ANSWER_AGENT_PORT:-8001}
VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv-cu124/bin/python}

cd "${VERL_HOME}"

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_PATH}" \
    --dtype bfloat16 \
    --gpu-memory-utilization "${ANSWER_AGENT_GPU_MEMORY_UTILIZATION:-0.6}" \
    --port "${PORT}"
