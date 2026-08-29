#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_PATH=${ANSWER_AGENT_MODEL:-Qwen/Qwen3-0.6B}
PORT=${ANSWER_AGENT_PORT:-8001}
VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv-cu124/bin/python}
GPU_MEMORY=${ANSWER_AGENT_GPU_MEMORY_UTILIZATION:-0.6}
MAX_MODEL_LEN=${ANSWER_AGENT_MAX_MODEL_LEN:-8192}

cd "${VERL_HOME}"

echo "Answer Agent: model=${MODEL_PATH}, gpu_memory=${GPU_MEMORY}, max_model_len=${MAX_MODEL_LEN}"

ARGS=(
    --model "${MODEL_PATH}"
    --served-model-name "${MODEL_PATH}"
    --dtype bfloat16
    --gpu-memory-utilization "${GPU_MEMORY}"
    --max-model-len "${MAX_MODEL_LEN}"
    --port "${PORT}"
)
exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
