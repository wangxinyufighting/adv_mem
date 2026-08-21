#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    exec bash "${ROOT}/scripts/run_alternating.sh" "$@"
fi

if [[ -f "${ROOT}/.env" ]]; then
    set -a
    source "${ROOT}/.env"
    set +a
fi

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in .env}"
: "${MOS_EMBEDDER_API_KEY:?Set MOS_EMBEDDER_API_KEY in .env}"
: "${MOS_EMBEDDER_API_BASE:?Set MOS_EMBEDDER_API_BASE in .env}"

TRAIN_GPUS=${TRAIN_GPUS:-0}
ANSWER_AGENT_GPU=${ANSWER_AGENT_GPU:-1}
RERANKER_GPU=${RERANKER_GPU:-2}
START_ANSWER_AGENT=${START_ANSWER_AGENT:-1}
START_RERANKER=${START_RERANKER:-1}
ANSWER_AGENT_PORT=${ANSWER_AGENT_PORT:-8001}
BGE_RERANKER_PORT=${BGE_RERANKER_PORT:-8000}

export ANSWER_AGENT_API_BASE=${ANSWER_AGENT_API_BASE:-http://127.0.0.1:${ANSWER_AGENT_PORT}/v1}
export BGE_RERANKER_URL=${BGE_RERANKER_URL:-http://127.0.0.1:${BGE_RERANKER_PORT}/v1/rerank}

LOG_DIR=${TRAIN_LOG_DIR:-${ROOT}/data/training/services}
mkdir -p "${LOG_DIR}"
PIDS=()

(
    cd "${ROOT}/verl"
    uv sync --frozen --all-packages --extra vllm --extra fsdp
)

cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

wait_for_url() {
    local url=$1
    for _ in $(seq 1 150); do
        curl -fsS "${url}" >/dev/null 2>&1 && return
        sleep 2
    done
    echo "Service did not start: ${url}" >&2
    exit 1
}

if [[ ${START_ANSWER_AGENT} == 1 ]]; then
    CUDA_VISIBLE_DEVICES=${ANSWER_AGENT_GPU} \
        bash "${ROOT}/scripts/serve_answer_agent.sh" \
        >"${LOG_DIR}/answer_agent.log" 2>&1 &
    PIDS+=("$!")
fi

if [[ ${START_RERANKER} == 1 ]]; then
    CUDA_VISIBLE_DEVICES=${RERANKER_GPU} \
        bash "${ROOT}/scripts/serve_bge_reranker.sh" \
        >"${LOG_DIR}/reranker.log" 2>&1 &
    PIDS+=("$!")
fi

wait_for_url "${ANSWER_AGENT_API_BASE%/}/models"
BGE_HEALTH_URL=${BGE_RERANKER_HEALTH_URL:-${BGE_RERANKER_URL%/v1/rerank}/v1/models}
wait_for_url "${BGE_HEALTH_URL}"

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} \
    bash "${ROOT}/scripts/run_alternating.sh" "$@"
