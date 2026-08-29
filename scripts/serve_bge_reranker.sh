#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_PATH=${BGE_RERANKER_MODEL_PATH:-BAAI/bge-reranker-v2-m3}
MODEL_NAME=${BGE_RERANKER_MODEL:-bge-reranker-v2-m3}
PORT=${BGE_RERANKER_PORT:-8000}
VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv-cu124/bin/python}
GPU_MEMORY=${RERANKER_GPU_MEMORY_UTILIZATION:-0.3}
MAX_MODEL_LEN=${RERANKER_MAX_MODEL_LEN:-2048}

cd "${VERL_HOME}"

echo "BGE Reranker: model=${MODEL_PATH}, gpu_memory=${GPU_MEMORY}, max_model_len=${MAX_MODEL_LEN}"

ARGS=(
    --model "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --task score
    --dtype bfloat16
    --gpu-memory-utilization "${GPU_MEMORY}"
    --max-model-len "${MAX_MODEL_LEN}"
    --port "${PORT}"
)
exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
