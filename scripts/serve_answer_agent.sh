#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_PATH=${ANSWER_AGENT_MODEL:-Qwen/Qwen3-0.6B}
PORT=${ANSWER_AGENT_PORT:-8001}

cd "${ROOT}/verl"

exec uv run --frozen --all-packages --extra vllm --extra fsdp python3 \
    -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_PATH}" \
    --port "${PORT}"
