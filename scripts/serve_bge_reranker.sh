#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_PATH=${BGE_RERANKER_MODEL_PATH:-BAAI/bge-reranker-v2-m3}
MODEL_NAME=${BGE_RERANKER_MODEL:-bge-reranker-v2-m3}
PORT=${BGE_RERANKER_PORT:-8000}

cd "${ROOT}/verl"

exec uv run --frozen --all-packages --extra vllm --extra fsdp python3 \
    -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --runner pooling \
    --port "${PORT}"
