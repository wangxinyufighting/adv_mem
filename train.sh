#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    exec bash "${ROOT}/scripts/run_alternating.sh" "$@"
fi

if [[ $(uname -s) != "Linux" || $(uname -m) != "x86_64" ]]; then
    echo "Training requires Linux x86_64 with CUDA; current platform is $(uname -s) $(uname -m)." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi is required." >&2
    exit 1
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
if (( ${DRIVER_VERSION%%.*} < 525 )); then
    echo "NVIDIA driver ${DRIVER_VERSION} is too old for CUDA 12.4; driver 525+ is required." >&2
    exit 1
fi

if ! command -v curl >/dev/null; then
    echo "curl is required." >&2
    exit 1
fi

if [[ -f "${ROOT}/.env" ]]; then
    set -a
    source "${ROOT}/.env"
    set +a
fi

export CUDA124_ENV_DIR=${CUDA124_ENV_DIR:-${ROOT}/.venv-cu124}
export VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
if [[ ! -x "${CUDA124_ENV_DIR}/bin/python" ]]; then
    bash "${ROOT}/scripts/setup_cuda124.sh"
fi
bash "${ROOT}/scripts/apply_verl_patch.sh"
export PYTHON_BIN=${CUDA124_ENV_DIR}/bin/python

if ! "${PYTHON_BIN}" -c 'import pkg_resources' >/dev/null 2>&1; then
    echo "Installing Verl-compatible setuptools..."
    "${PYTHON_BIN}" -m pip install "setuptools==80.9.0"
fi

"${PYTHON_BIN}" -c \
    'import flash_attn, torch, vllm; assert torch.version.cuda == "12.4", f"Expected CUDA 12.4, got {torch.version.cuda}"; assert vllm.__version__ == "0.8.5.post1", vllm.__version__'
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
SERVICE_START_TIMEOUT=${SERVICE_START_TIMEOUT:-900}

port_is_free() {
    "${PYTHON_BIN}" -c \
        'import socket, sys; s = socket.socket(); s.bind(("0.0.0.0", int(sys.argv[1]))); s.close()' \
        "$1" 2>/dev/null
}

free_port() {
    "${PYTHON_BIN}" -c \
        'import socket; s = socket.socket(); s.bind(("0.0.0.0", 0)); print(s.getsockname()[1]); s.close()'
}

if [[ ${START_ANSWER_AGENT} == 1 ]]; then
    if ! port_is_free "${ANSWER_AGENT_PORT}"; then
        old_port=${ANSWER_AGENT_PORT}
        ANSWER_AGENT_PORT=$(free_port)
        echo "Port ${old_port} is busy; Answer Agent will use ${ANSWER_AGENT_PORT}."
    fi
    export ANSWER_AGENT_API_BASE=http://127.0.0.1:${ANSWER_AGENT_PORT}/v1
else
    export ANSWER_AGENT_API_BASE=${ANSWER_AGENT_API_BASE:-http://127.0.0.1:${ANSWER_AGENT_PORT}/v1}
fi

if [[ ${START_RERANKER} == 1 ]]; then
    if ! port_is_free "${BGE_RERANKER_PORT}"; then
        old_port=${BGE_RERANKER_PORT}
        BGE_RERANKER_PORT=$(free_port)
        echo "Port ${old_port} is busy; BGE Reranker will use ${BGE_RERANKER_PORT}."
    fi
    export BGE_RERANKER_URL=http://127.0.0.1:${BGE_RERANKER_PORT}/v1/rerank
else
    export BGE_RERANKER_URL=${BGE_RERANKER_URL:-http://127.0.0.1:${BGE_RERANKER_PORT}/v1/rerank}
fi

if [[ ${ANSWER_AGENT_PORT} == "${BGE_RERANKER_PORT}" ]]; then
    echo "Answer Agent and BGE Reranker must use different ports." >&2
    exit 1
fi

export ANSWER_AGENT_PORT BGE_RERANKER_PORT

WORK_DIR=${ROOT}/data/training
ARGS=("$@")
for ((index = 0; index < ${#ARGS[@]}; index++)); do
    case ${ARGS[index]} in
        --work-dir) WORK_DIR=${ARGS[index + 1]} ;;
        --work-dir=*) WORK_DIR=${ARGS[index]#*=} ;;
    esac
done
[[ ${WORK_DIR} == /* ]] || WORK_DIR=${ROOT}/${WORK_DIR}

LOG_DIR=${TRAIN_LOG_DIR:-${WORK_DIR}/services}
mkdir -p "${LOG_DIR}"
PIDS=()

cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

wait_for_service() {
    local name=$1
    local url=$2
    local pid=$3
    local log=$4
    for ((elapsed = 0; elapsed < SERVICE_START_TIMEOUT; elapsed += 2)); do
        curl -fsS "${url}" >/dev/null 2>&1 && return
        if [[ -n ${pid} ]] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "${name} exited during startup." >&2
            [[ -f ${log} ]] && tail -n 80 "${log}" >&2
            exit 1
        fi
        sleep 2
    done
    echo "${name} did not start: ${url}" >&2
    [[ -f ${log} ]] && tail -n 80 "${log}" >&2
    exit 1
}

ANSWER_LOG=${LOG_DIR}/answer_agent.log
RERANKER_LOG=${LOG_DIR}/reranker.log
ANSWER_PID=""
RERANKER_PID=""
SHARED_SERVICE_GPU=0
if [[ ${START_ANSWER_AGENT} == 1 && ${START_RERANKER} == 1 && \
      ${ANSWER_AGENT_GPU} == "${RERANKER_GPU}" ]]; then
    SHARED_SERVICE_GPU=1
fi

if [[ ${START_ANSWER_AGENT} == 1 ]]; then
    echo "Starting Answer Agent on GPU ${ANSWER_AGENT_GPU}..."
    CUDA_VISIBLE_DEVICES=${ANSWER_AGENT_GPU} \
        bash "${ROOT}/scripts/serve_answer_agent.sh" \
        >"${ANSWER_LOG}" 2>&1 &
    ANSWER_PID=$!
    PIDS+=("${ANSWER_PID}")
fi

# Avoid two vLLM processes measuring and allocating the same GPU concurrently.
if [[ ${SHARED_SERVICE_GPU} == 1 ]]; then
    wait_for_service \
        "Answer Agent" \
        "${ANSWER_AGENT_API_BASE%/}/models" \
        "${ANSWER_PID}" \
        "${ANSWER_LOG}"
fi

if [[ ${START_RERANKER} == 1 ]]; then
    echo "Starting BGE Reranker on GPU ${RERANKER_GPU}..."
    CUDA_VISIBLE_DEVICES=${RERANKER_GPU} \
        bash "${ROOT}/scripts/serve_bge_reranker.sh" \
        >"${RERANKER_LOG}" 2>&1 &
    RERANKER_PID=$!
    PIDS+=("${RERANKER_PID}")
fi

if [[ ${SHARED_SERVICE_GPU} == 0 ]]; then
    wait_for_service \
        "Answer Agent" \
        "${ANSWER_AGENT_API_BASE%/}/models" \
        "${ANSWER_PID}" \
        "${ANSWER_LOG}"
fi
BGE_HEALTH_URL=${BGE_RERANKER_HEALTH_URL:-${BGE_RERANKER_URL%/v1/rerank}/v1/models}
wait_for_service \
    "BGE Reranker" \
    "${BGE_HEALTH_URL}" \
    "${RERANKER_PID}" \
    "${RERANKER_LOG}"

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} \
    bash "${ROOT}/scripts/run_alternating.sh" "$@" \
    2>&1 | tee "${LOG_DIR}/training.log"
