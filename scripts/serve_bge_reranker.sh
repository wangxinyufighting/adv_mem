#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_PATH=${BGE_RERANKER_MODEL_PATH:-BAAI/bge-reranker-v2-m3}
MODEL_NAME=${BGE_RERANKER_MODEL:-bge-reranker-v2-m3}
PORT=${BGE_RERANKER_PORT:-8000}
VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv-cu124/bin/python}

cd "${VERL_HOME}"

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --task score \
    --gpu-memory-utilization "${RERANKER_GPU_MEMORY_UTILIZATION:-0.3}" \
    --port "${PORT}"
